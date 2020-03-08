# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An LSTM cell compatible with the Haste LayerNormLSTM layer."""


import tensorflow as tf

from tensorflow.compat import v1
from tensorflow.compat.v1.nn import rnn_cell


__all__ = [
    'LayerNormLSTMCell'
]


class LayerNormLSTMCell(rnn_cell.RNNCell):
  """
  An LSTM cell that's compatible with the Haste LayerNormLSTM layer.

  This cell can be used on hardware other than GPUs and with other TensorFlow
  classes that operate on RNN cells (e.g. `dynamic_rnn`, `BasicDecoder`, cell
  wrappers, etc.).
  """

  def __init__(self,
        num_units,
        forget_bias=1.0,
        dropout=0.0,
        dtype=None,
        name=None,
        **kwargs):
    super(LayerNormLSTMCell, self).__init__(dtype=dtype, name=name, **kwargs)
    self.realname = name
    self.num_units = num_units

    self.forget_bias = forget_bias
    self.dropout = dropout
    self.kernel = None
    self.recurrent_kernel = None
    self.bias = None
    self.gamma = None
    self.gamma_h = None
    self.beta_h = None
    self.built = False

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(self.num_units, self.num_units)

  @property
  def output_size(self):
    return self.num_units

  def build(self, shape):
    num_units = self.num_units
    input_size = int(shape[-1])

    # No user-supplied initializers here since this class should only really
    # be used for inference on a pre-trained model.
    with tf.name_scope(self.name), v1.variable_scope(self.realname, 'lstm_cell'):
      self._kernel = v1.get_variable('kernel', shape=[input_size + num_units, num_units * 4])
      self.kernel, self.recurrent_kernel = tf.split(self._kernel, [input_size, num_units], axis=0)
      self.bias = v1.get_variable('bias', shape=[num_units * 4], initializer=v1.initializers.zeros())
      self.gamma = v1.get_variable('gamma', shape=[2, num_units * 4], initializer=v1.initializers.ones())
      self.gamma_h = v1.get_variable('gamma_h', shape=[num_units], initializer=v1.initializers.ones())
      self.beta_h = v1.get_variable('beta_h', shape=[num_units], initializer=v1.initializers.zeros())
      self.null = tf.zeros_like(self.gamma[0])
    self.built = True

  def __call__(self, inputs, state, scope=None):
    self.build(inputs.shape)

    R = tf.nn.dropout(self.recurrent_kernel, rate=self.dropout)

    Wx = self._layer_norm(tf.matmul(inputs, self.kernel), self.gamma[0], self.null)
    Rh = self._layer_norm(tf.matmul(state.h, R), self.gamma[1], self.null)
    v = Wx + Rh + self.bias
    v_i, v_g, v_f, v_o = tf.split(v, 4, axis=-1)
    i = tf.nn.sigmoid(v_i)
    g = tf.nn.tanh   (v_g)
    f = tf.nn.sigmoid(v_f)
    o = tf.nn.sigmoid(v_o)
    c_new = f * state.c + i * g
    c_tanh = tf.nn.tanh(self._layer_norm(c_new, self.gamma_h, self.beta_h))
    h_new = o * c_tanh

    return h_new, rnn_cell.LSTMStateTuple(c_new, h_new)

  def _layer_norm(self, x, gamma, beta):
    mean, variance = tf.nn.moments(x, axes=[-1], keepdims=True)
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)

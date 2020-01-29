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

"""A GRU cell compatible with the Haste GRU layer."""


import tensorflow as tf

from tensorflow.compat import v1
from tensorflow.compat.v1.nn import rnn_cell


class GRUCell(rnn_cell.RNNCell):
  """
  A GRU cell that's compatible with the Haste GRU layer.

  This cell can be used on hardware other than GPUs and with other TensorFlow
  classes that operate on RNN cells (e.g. `dynamic_rnn`, `BasicDecoder`, cell
  wrappers, etc.).
  """
  def __init__(self, num_units, name=None, **kwargs):
    super(GRUCell, self).__init__(name=name, **kwargs)

    self.realname = name
    self.num_units = num_units
    self.built = False

  @property
  def state_size(self):
    return self.num_units

  @property
  def output_size(self):
    return self.num_units

  def build(self, shape):
    if self.built:
      return

    num_units = self.num_units
    input_size = int(shape[-1])
    dtype = self.dtype or tf.float32

    kernel_initializer = v1.initializers.glorot_uniform(dtype=dtype)
    bias_initializer = v1.initializers.zeros(dtype=dtype)

    with tf.name_scope(self.name), v1.variable_scope(self.realname, 'gru_cell'):
      self._kernel = v1.get_variable('kernel', initializer=lambda: kernel_initializer([input_size + num_units, num_units * 3]))
      self._bias = v1.get_variable('bias', initializer=lambda: bias_initializer([num_units * 6]))

    self.kernel, self.recurrent_kernel = tf.split(self._kernel, [input_size, num_units], axis=0)
    self.bias, self.recurrent_bias = tf.split(self._bias, 2, axis=0)

    self.built = True

  def __call__(self, inputs, state, scope=None):
    self.build(inputs.shape)

    h_proj = tf.nn.xw_plus_b(state, self.recurrent_kernel, self.recurrent_bias)
    x = tf.nn.xw_plus_b(inputs, self.kernel, self.bias)
    h_z, h_r, h_g = tf.split(h_proj, 3, axis=-1)
    x_z, x_r, x_g = tf.split(x, 3, axis=-1)
    z = tf.nn.sigmoid(h_z + x_z)
    r = tf.nn.sigmoid(h_r + x_r)
    g = tf.nn.tanh(r * h_g + x_g)
    h = z * state + (1 - z) * g
    return h, h

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

"""Long Short-Term Memory"""


import pkg_resources
import tensorflow as tf

from tensorflow.compat import v1
from tensorflow.compat.v1.nn import rnn_cell

from .base_rnn import BaseRNN
from .weight_config import WeightConfig


__all__ = [
    'LSTM'
]


LIB = tf.load_op_library(pkg_resources.resource_filename(__name__, 'libhaste_tf.so'))


@tf.RegisterGradient("HasteLstm")
def lstm_gradient(op, *grads):
  training = op.get_attr('training')
  if not training:
    raise ValueError(('LSTM can only compute gradients if `training=True` was specified during the '
                      'forward pass.\nFailed op: {}').format(op.name))

  # Extract inputs and outputs from the op.
  x = op.inputs[0]
  W = op.inputs[1]
  R = op.inputs[2]
  b = op.inputs[3]
  zoneout_mask = op.inputs[4]
  h = op.outputs[0]
  c = op.outputs[1]
  v = op.outputs[2]

  # Pre-transpose matrices for better performance.
  x = tf.transpose(x, [2, 0, 1])
  W = tf.transpose(W, [1, 0])
  R = tf.transpose(R, [1, 0])

  dx, dW, dR, db = LIB.haste_lstm_grad(x, W, R, b, h, c, v, grads[0], grads[1], zoneout_mask)
  return [dx, dW, dR, db, None]


class LSTMLayer(tf.Module):
  def __init__(self,
        num_units,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        kernel_transform=None,
        recurrent_transform=None,
        bias_transform=None,
        forget_bias=1.0,
        dropout=0.0,
        zoneout=0.0,
        dtype=None,
        name=None,
        cudnn_compat=False):
    super(LSTMLayer, self).__init__(name)
    self.realname = name
    self.input_size = None
    self.num_units = num_units

    identity = lambda x: x
    self.kernel_config = WeightConfig(v1.initializers.glorot_uniform(), None, identity)
    self.recurrent_config = WeightConfig(v1.initializers.orthogonal(), None, identity)
    self.bias_config = WeightConfig(v1.initializers.zeros(), None, identity)

    self.kernel_config.override(kernel_initializer, None, kernel_transform)
    self.recurrent_config.override(recurrent_initializer, None, recurrent_transform)
    self.bias_config.override(bias_initializer, None, bias_transform)

    self.forget_bias = forget_bias
    self.dropout = dropout
    self.zoneout = zoneout
    self.dtype = dtype or tf.float32
    self.cudnn_compat = cudnn_compat
    self.opaque = None
    self.kernel = None
    self.bias = None
    self.built = False

  def build(self, shape):
    if self.built:
      return

    num_units = self.num_units
    input_size = int(shape[-1])

    kernel_shape = tf.TensorShape([input_size, num_units])
    recurrent_shape = tf.TensorShape([num_units, num_units])
    bias_shape = tf.TensorShape([num_units])

    kernel_weights = [self.kernel_config.initializer(kernel_shape, dtype=self.dtype) for _ in range(4)]
    recurrent_weights = [self.recurrent_config.initializer(recurrent_shape, dtype=self.dtype) for _ in range(4)]
    if self.forget_bias:
      biases = [tf.zeros(bias_shape, dtype=self.dtype) for _ in range(4)]
      biases[2] = tf.constant(self.forget_bias, shape=bias_shape, dtype=self.dtype)
    else:
      biases = [self.bias_config.initializer(bias_shape, dtype=self.dtype) for _ in range(4)]

    kernel_weights = tf.concat(kernel_weights, axis=-1)
    recurrent_weights = tf.concat(recurrent_weights, axis=-1)
    biases = tf.concat(biases, axis=-1)

    if not self.cudnn_compat:
      # Use the same format as LSTMBlockCell.
      with self.name_scope, v1.variable_scope(self.realname, 'lstm_cell'):
        weights = tf.concat([kernel_weights, recurrent_weights], axis=0)
        self.kernel = v1.get_variable('kernel', initializer=weights)
        self.bias = v1.get_variable('bias', initializer=biases)
    else:
      # Use the same format as CudnnLSTM.
      with self.name_scope, v1.variable_scope(self.realname, 'lstm_cell'):
        with v1.variable_scope('cudnn_lstm'):
          # Sigh, cuDNN uses two bias vectors instead of just one.
          extra_biases = [self.bias_initializer(tf.TensorShape([num_units]), dtype=self.dtype) for _ in range(4)]
          extra_biases = tf.concat(extra_biases, axis=-1)
          kernel_weights = tf.reshape(kernel_weights, [-1])
          recurrent_weights = tf.reshape(recurrent_weights, [-1])
          opaque_initial_value = tf.concat([kernel_weights, recurrent_weights, biases, extra_biases], axis=-1)
          self.opaque = v1.get_variable('opaque_kernel', initializer=opaque_initial_value)

    self.input_size = input_size
    self.built = True

  def get_weights(self):
    if self.cudnn_compat:
      # Split into 3 variables.
      W_size = 4 * self.input_size * self.num_units
      R_size = 4 * self.num_units * self.num_units
      b_size = 8 * self.num_units
      kernel, recurrent_kernel, bias = tf.split(opaque, [W_size, R_size, b_size])

      # Convert from cuDNN [i, f, g, o] format to TF and LMNT [i, g, f, o] format.
      # Note that we only use a single bias vector so we sum the two separate ones
      # and then reorder formats.
      Wi, Wf, Wg, Wo = tf.split(kernel, 4)
      Ri, Rf, Rg, Ro = tf.split(recurrent_kernel, 4)
      bi, bf, bg, bo = tf.split(tf.reduce_sum(tf.split(bias, 2), axis=0), 4)
      kernel = tf.concat([Wi, Wg, Wf, Wo], axis=0)
      recurrent_kernel = tf.concat([Ri, Rg, Rf, Ro], axis=0)
      bias = tf.concat([bi, bg, bf, bo], axis=0)

      # Shape them correctly.
      kernel = tf.reshape(kernel, [4 * self.num_units, self.input_size])
      recurrent_kernel = tf.reshape(recurrent_kernel, [4 * self.num_units, self.num_units])
      bias = tf.reshape(bias, [4 * self.num_units])

      # Pre-transpose the kernels.
      kernel = tf.transpose(kernel, [1, 0])
      recurrent_kernel = tf.transpose(recurrent_kernel, [1, 0])
    else:
      kernel = self.kernel[:-self.num_units]
      recurrent_kernel = self.kernel[-self.num_units:]
      bias = self.bias
    return {
        'kernel': self.kernel_config.transform(kernel),
        'recurrent_kernel': self.recurrent_config.transform(recurrent_kernel),
        'bias': self.bias_config.transform(bias)
    }

  @property
  def state_size(self):
    return rnn_cell.LSTMStateTuple(self.num_units, self.num_units)

  @property
  def output_size(self):
    return self.num_units

  def __call__(self, x, sequence_length, training):
    self.build(x.shape)

    shape = tf.shape(x)
    time_steps = shape[0]
    batch_size = shape[1]

    # Use an empty zoneout mask if no zoneout is going to be applied.
    # Sadly, we can't pass `None` to the op but at least we won't be wasting
    # memory or bandwidth on this tensor.
    zoneout_mask = tf.zeros([0, 0, 0], dtype=self.dtype)
    if self.zoneout:
      zoneout_mask = 1.0 - self.zoneout
      zoneout_mask += tf.random.uniform([time_steps, batch_size, self.num_units], dtype=self.dtype)
      zoneout_mask = tf.floor(zoneout_mask)

    weights = self.get_weights()
    if training and self.dropout > 0:
      recurrent_kernel = tf.nn.dropout(weights['recurrent_kernel'], rate=self.dropout)
    else:
      recurrent_kernel = weights['recurrent_kernel']
    h, c, _ = LIB.haste_lstm(
        x,
        weights['kernel'],
        recurrent_kernel,
        weights['bias'],
        zoneout_mask,
        training=training,
        zoneout_prob=self.zoneout)

    if sequence_length is not None:
      indices = sequence_length
      indices = tf.stack([indices, tf.range(batch_size, dtype=sequence_length.dtype)], axis=-1)
      state = rnn_cell.LSTMStateTuple(tf.gather_nd(c, indices), tf.gather_nd(h, indices))
    else:
      state = rnn_cell.LSTMStateTuple(c[-1], h[-1])

    return h[1:], state


class LSTM(BaseRNN):
  """
  Long Short-Term Memory layer.

  This LSTM layer offers a fused, GPU-accelerated TensorFlow op for inference
  and training. Its weights and variables are compatible with `BasicLSTMCell`,
  `LSTMCell`, and `LSTMBlockCell` by default, and is able to load weights
  from `tf.contrib.cudnn_rnn.CudnnLSTM` when `cudnn_compat=True` is specified.

  Although this implementation is comparable in performance to cuDNN's LSTM,
  it offers additional options not typically found in other high-performance
  implementations. DropConnect and Zoneout regularization are built-in, and
  this layer allows setting a non-zero initial forget gate bias.
  """

  def __init__(self, num_units, direction='unidirectional', **kwargs):
    """
    Initialize the parameters of the LSTM layer.

    Arguments:
      num_units: int, the number of units in the LSTM cell.
      direction: string, 'unidirectional' or 'bidirectional'.
      **kwargs: Dict, keyword arguments (see below).

    Keyword Arguments:
      kernel_initializer: (optional) the initializer to use for the input
        matrix weights. Defaults to `glorot_uniform`.
      recurrent_initializer: (optional) the initializer to use for the
        recurrent matrix weights. Defaults to `orthogonal`.
      bias_initializer: (optional) the initializer to use for both input and
        recurrent bias vectors. Defaults to `zeros` unless `forget_bias` is
        non-zero (see below).
      kernel_transform: (optional) a function with signature
        `(kernel: Tensor) -> Tensor` that transforms the kernel before it is
        used. Defaults to the identity function.
      recurrent_transform: (optional) a function with signature
        `(recurrent_kernel: Tensor) -> Tensor` that transforms the recurrent
        kernel before it is used. Defaults to the identity function.
      bias_transform: (optional) a function with signature
        `(bias: Tensor) -> Tensor` that transforms the bias before it is used.
        Defaults to the identity function.
      forget_bias: (optional) float, sets the initial weights for the forget
        gates. Defaults to 1 and overrides the `bias_initializer` unless this
        argument is set to 0.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix. Defaults to 0.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization. Defaults to 0.
      dtype: (optional) the data type for this layer. Defaults to `tf.float32`.
      name: (optional) string, the name for this layer.
      cudnn_compat: (optional) bool, if `True`, the variables created by this
        layer are compatible with `tf.contrib.cudnn_rnn.CudnnLSTM`. Note that
        this should only be set if you're restoring variables from a cuDNN
        model. It's currently not possible to train a model with
        `cudnn_compat=True` and restore it with CudnnLSTM. Defaults to `False`.
    """
    super().__init__(LSTMLayer, num_units, direction, 'lstm_cell', **kwargs)

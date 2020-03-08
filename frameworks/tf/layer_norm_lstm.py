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

"""Layer Normalized Long Short-Term Memory"""


import pkg_resources
import tensorflow as tf

from tensorflow.compat import v1
from tensorflow.compat.v1.nn import rnn_cell


__all__ = [
    'LayerNormLSTM'
]


LIB = tf.load_op_library(pkg_resources.resource_filename(__name__, 'libhaste_tf.so'))


def reverse_sequence(sequence, sequence_length):
  """
  Reverses a batched sequence in time-major order [T,N,...]. The input sequence
  may be padded, in which case sequence_length specifies the unpadded length of
  each sequence.
  """
  if sequence_length is None:
    return tf.reverse(sequence, axis=[0])
  return tf.reverse_sequence(sequence, sequence_length, seq_axis=0, batch_axis=1)


def transpose(tensor_or_tuple, perm):
  """Transposes the given tensor or tuple of tensors by the same permutation."""
  if isinstance(tensor_or_tuple, tuple):
    return tuple([tf.transpose(tensor, perm) for tensor in tensor_or_tuple])
  return tf.transpose(tensor_or_tuple, perm)


@tf.RegisterGradient("HasteLayerNormLstm")
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
  gamma = op.inputs[4]
  gamma_h = op.inputs[5]
  beta_h = op.inputs[6]
  zoneout_mask = op.inputs[7]
  h = op.outputs[0]
  c = op.outputs[1]
  cache = op.outputs[2]

  # Pre-transpose matrices for better performance.
  x = tf.transpose(x, [2, 0, 1])
  W = tf.transpose(W, [1, 0])
  R = tf.transpose(R, [1, 0])

  dx, dW, dR, db, dgamma, dgamma_h, dbeta_h = LIB.haste_layer_norm_lstm_grad(
      x,
      W,
      R,
      b,
      gamma,
      gamma_h,
      beta_h,
      h,
      c,
      cache,
      grads[0],
      grads[1],
      zoneout_mask)
  return [dx, dW, dR, db, dgamma, dgamma_h, dbeta_h, None]


class LayerNormLSTMLayer(tf.Module):
  def __init__(self,
        num_units,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        forget_bias=1.0,
        dropout=0.0,
        zoneout=0.0,
        dtype=None,
        name=None):
    super(LayerNormLSTMLayer, self).__init__(name)
    self.realname = name
    self.num_units = num_units

    self.kernel_initializer = kernel_initializer or v1.initializers.glorot_uniform()
    self.recurrent_initializer = recurrent_initializer or v1.initializers.orthogonal()
    self.bias_initializer = bias_initializer or v1.initializers.zeros()

    self.forget_bias = forget_bias
    self.dropout = dropout
    self.zoneout = zoneout
    self.dtype = dtype or tf.float32
    self.kernel = None
    self.recurrent_kernel = None
    self.bias = None
    self.gamma = None
    self.gamma_h = None
    self.beta_h = None
    self.built = False

  def build(self, shape):
    if self.built:
      return

    num_units = self.num_units
    input_size = int(shape[-1])

    kernel_shape = tf.TensorShape([input_size, num_units])
    recurrent_shape = tf.TensorShape([num_units, num_units])
    bias_shape = tf.TensorShape([num_units])

    kernel_weights = [self.kernel_initializer(kernel_shape, dtype=self.dtype) for _ in range(4)]
    recurrent_weights = [self.recurrent_initializer(recurrent_shape, dtype=self.dtype) for _ in range(4)]
    if self.forget_bias:
      biases = [tf.zeros(bias_shape, dtype=self.dtype) for _ in range(4)]
      biases[2] = tf.constant(self.forget_bias, shape=bias_shape, dtype=self.dtype)
    else:
      biases = [self.bias_initializer(bias_shape, dtype=self.dtype) for _ in range(4)]

    kernel_weights = tf.concat(kernel_weights, axis=-1)
    recurrent_weights = tf.concat(recurrent_weights, axis=-1)
    biases = tf.concat(biases, axis=-1)

    # Use the same format as LSTMBlockCell.
    with self.name_scope, v1.variable_scope(self.realname, 'lstm_cell'):
      weights = tf.concat([kernel_weights, recurrent_weights], axis=0)
      self._kernel = v1.get_variable('kernel', initializer=weights)
      self.kernel, self.recurrent_kernel = tf.split(self._kernel, [input_size, num_units], axis=0)
      self.bias = v1.get_variable('bias', initializer=biases)
      self.gamma = v1.get_variable('gamma', shape=[2, self.num_units * 4], initializer=v1.initializers.ones())
      self.gamma_h = v1.get_variable('gamma_h', shape=[self.num_units], initializer=v1.initializers.ones())
      self.beta_h = v1.get_variable('beta_h', shape=[self.num_units], initializer=v1.initializers.zeros())
    self.built = True

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
      zoneout_mask += tf.random_uniform([time_steps, batch_size, self.num_units], dtype=self.dtype)
      zoneout_mask = tf.floor(zoneout_mask)

    recurrent_kernel = tf.nn.dropout(self.recurrent_kernel, rate=self.dropout)
    h, c, _ = LIB.haste_layer_norm_lstm(
        x,
        self.kernel,
        recurrent_kernel,
        self.bias,
        self.gamma,
        self.gamma_h,
        self.beta_h,
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


class LayerNormLSTM(tf.Module):
  """
  Layer Normalized Long Short-Term Memory layer.

  This LSTM layer applies layer normalization to the input, recurrent, and
  output activations of a standard LSTM. The implementation is fused and
  GPU-accelerated. DropConnect and Zoneout regularization are built-in, and
  this layer allows setting a non-zero initial forget gate bias.

  Details about the exact function this layer implements can be found at
  https://github.com/lmnt-com/haste/issues/1.
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
      forget_bias: (optional) float, sets the initial weights for the forget
        gates. Defaults to 1 and overrides the `bias_initializer` unless this
        argument is set to 0.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix. Defaults to 0.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization. Defaults to 0.
      dtype: (optional) the data type for this layer. Defaults to `tf.float32`.
      name: (optional) string, the name for this layer.
    """
    assert direction in ['unidirectional', 'bidirectional']

    if direction == 'bidirectional':
      name = kwargs.pop('name', None)
      super(LayerNormLSTM, self).__init__(name)
      self.realname = name
      self.fwd_lstm = LayerNormLSTMLayer(num_units, name='fw', **kwargs)
      self.bwd_lstm = LayerNormLSTMLayer(num_units, name='bw', **kwargs)
    else:
      super(LayerNormLSTM, self).__init__()
      self.fwd_lstm = LayerNormLSTMLayer(num_units, **kwargs)
      self.bwd_lstm = None

  def build(self, shape):
    """
    Creates the variables of the layer.

    Calling this method is optional for users of the LSTM class. It is called
    internally with the correct shape when `__call__` is invoked.

    Arguments:
      shape: instance of `TensorShape`.
    """
    if self.bwd_lstm is not None:
      with self.name_scope, v1.variable_scope(self.realname, 'lstm_cell'):
        self.fwd_lstm.build(shape)
        self.bwd_lstm.build(shape)
    else:
      self.fwd_lstm.build(shape)

  @property
  def output_size(self):
    if self.bwd_lstm is not None:
      return self.fwd_lstm.output_size, self.bwd_lstm.output_size
    return self.fwd_lstm.output_size

  @property
  def state_size(self):
    if self.bwd_lstm is not None:
      return self.fwd_lstm.state_size, self.bwd_lstm.state_size
    return self.fwd_lstm.state_size

  def __call__(self, inputs, training, sequence_length=None, time_major=False):
    """
    Runs the LSTM layer.

    Arguments:
      inputs: Tensor, a rank 3 input tensor with shape [N,T,C] if `time_major`
        is `False`, or with shape [T,N,C] if `time_major` is `True`.
      training: bool, `True` if running in training mode, `False` if running
        in inference mode.
      sequence_length: (optional) Tensor, a rank 1 tensor with shape [N] and
        dtype of `tf.int32` or `tf.int64`. This tensor specifies the unpadded
        length of each example in the input minibatch.
      time_major: (optional) bool, specifies whether `input` has shape [N,T,C]
        (`time_major=False`) or shape [T,N,C] (`time_major=True`).

    Returns:
      A pair, `(output, state)` for unidirectional layers, or a pair
      `([output_fwd, output_bwd], [state_fwd, state_bwd])` for bidirectional
      layers. Each state object will be an instance of `LSTMStateTuple`.
    """
    self.build(inputs.shape)

    if not time_major:
      inputs = transpose(inputs, [1, 0, 2])

    result, state = self.fwd_lstm(inputs, sequence_length, training)

    if self.bwd_lstm is not None:
      inputs = reverse_sequence(inputs, sequence_length)
      bwd_result, bwd_state = self.bwd_lstm(inputs, sequence_length, training)
      result = result, reverse_sequence(bwd_result, sequence_length)
      state = state, bwd_state

    if not time_major:
      result = transpose(result, [1, 0, 2])

    return result, state

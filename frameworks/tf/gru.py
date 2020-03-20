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

"""Gated Recurrent Unit"""


import pkg_resources
import tensorflow as tf

from tensorflow.compat import v1


__all__ = [
    'GRU'
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


@tf.RegisterGradient("HasteGru")
def gru_gradient(op, *grads):
  training = op.get_attr('training')
  if not training:
    raise ValueError(('GRU can only compute gradients if `training=True` was specified during the '
                      'forward pass.\nFailed op: {}').format(op.name))

  # Extract inputs and outputs from the op.
  x = op.inputs[0]
  W = op.inputs[1]
  R = op.inputs[2]
  bx = op.inputs[3]
  br = op.inputs[4]
  zoneout_mask = op.inputs[5]
  h = op.outputs[0]
  v = op.outputs[1]

  # Pre-transpose matrices for better performance.
  x = tf.transpose(x, [2, 0, 1])
  W = tf.transpose(W, [1, 0])
  R = tf.transpose(R, [1, 0])

  dx, dW, dR, dbx, dbr = LIB.haste_gru_grad(x, W, R, bx, br, h, v, grads[0], zoneout_mask)

  return [dx, dW, dR, dbx, dbr, None]


class GRULayer(tf.Module):
  def __init__(self,
        num_units,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        dropout=0.0,
        zoneout=0.0,
        dtype=None,
        name=None):
    super(GRULayer, self).__init__(name)
    self.realname = name
    self.num_units = num_units

    self.kernel_initializer = kernel_initializer or v1.initializers.glorot_uniform()
    self.recurrent_initializer = recurrent_initializer or v1.initializers.orthogonal()
    self.bias_initializer = bias_initializer or v1.initializers.zeros()

    self.dropout = dropout
    self.zoneout = zoneout
    self.dtype = dtype or tf.float32
    self.kernel = None
    self.recurrent_kernel = None
    self.bias = None
    self.recurrent_bias = None
    self.built = False

  def build(self, shape):
    if self.built:
      return

    num_units = self.num_units
    input_size = int(shape[-1])

    kernel_shape = tf.TensorShape([input_size, num_units])
    recurrent_shape = tf.TensorShape([num_units, num_units])
    bias_shape = tf.TensorShape([num_units])
    recurrent_bias_shape = tf.TensorShape([num_units])

    kernel_weights = [self.kernel_initializer(kernel_shape, dtype=self.dtype) for _ in range(3)]
    recurrent_weights = [self.recurrent_initializer(recurrent_shape, dtype=self.dtype) for _ in range(3)]
    biases = [self.bias_initializer(bias_shape) for _ in range(3)]
    recurrent_biases = [self.bias_initializer(recurrent_bias_shape) for _ in range(3)]

    kernel_weights = tf.concat(kernel_weights, axis=-1)
    recurrent_weights = tf.concat(recurrent_weights, axis=-1)
    biases = tf.concat(biases, axis=-1)
    recurrent_biases = tf.concat(recurrent_biases, axis=-1)

    weights = tf.concat([kernel_weights, recurrent_weights], axis=0)
    biases = tf.concat([biases, recurrent_biases], axis=0)

    with self.name_scope, v1.variable_scope(self.realname, 'gru_cell'):
      self._kernel = v1.get_variable('kernel', initializer=weights)
      self._bias = v1.get_variable('bias', initializer=biases)

    self.kernel, self.recurrent_kernel = tf.split(self._kernel, [input_size, num_units], axis=0)
    self.bias, self.recurrent_bias = tf.split(self._bias, 2, axis=0)

    self.built = True

  def __call__(self, inputs, sequence_length, training):
    self.build(inputs.shape)

    shape = tf.shape(inputs)
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
    result, _ = LIB.haste_gru(
        inputs,
        self.kernel,
        recurrent_kernel,
        self.bias,
        self.recurrent_bias,
        zoneout_mask,
        training=training,
        zoneout_prob=self.zoneout)

    if sequence_length is not None:
      # 0-indexed tensors, so length-1.
      indices = sequence_length
      indices = tf.stack([indices, tf.range(batch_size, dtype=sequence_length.dtype)], axis=-1)
      state = tf.gather_nd(result, indices)
    else:
      state = result[-1]

    return result[1:], state


class GRU(tf.Module):
  """
  Gated Recurrent Unit layer.

  This GRU layer offers a fused, GPU-accelerated TensorFlow op for inference
  and training. There are two commonly-used variants of GRU cells. This one
  implements 1406.1078v1 which applies the reset gate to the hidden state
  after matrix multiplication. cuDNN also implements this variant. The other
  variant, 1406.1078v3, applies the reset gate before matrix multiplication
  and is currently unsupported.

  This layer has built-in support for DropConnect and Zoneout, which are
  both techniques used to regularize RNNs.
  """

  def __init__(self, num_units, direction='unidirectional', **kwargs):
    """
    Initialize the parameters of the GRU layer.

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
      super(GRU, self).__init__(name)
      self.realname = name
      self.fwd_gru = GRULayer(num_units, name='fw', **kwargs)
      self.bwd_gru = GRULayer(num_units, name='bw', **kwargs)
    else:
      super(GRU, self).__init__()
      self.fwd_gru = GRULayer(num_units, **kwargs)
      self.bwd_gru = None

  def build(self, shape):
    """
    Creates the variables of the layer.

    Calling this method is optional for users of the GRU class. It is called
    internally with the correct shape when `__call__` is invoked.

    Arguments:
        shape: instance of `TensorShape`.
    """
    if self.bwd_gru is not None:
      with self.name_scope, v1.variable_scope(self.realname, 'gru_cell'):
        self.fwd_gru.build(shape)
        self.bwd_gru.build(shape)
    else:
      self.fwd_gru.build(shape)

  @property
  def output_size(self):
    if self.bwd_gru is not None:
      return self.fwd_gru.output_size, self.bwd_gru.output_size
    return self.fwd_gru.output_size

  @property
  def state_size(self):
    if self.bwd_gru is not None:
      return self.fwd_gru.state_size, self.bwd_gru.state_size
    return self.fwd_gru.state_size

  def __call__(self, inputs, training, sequence_length=None, time_major=False):
    """
    Runs the GRU layer.

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
      layers.
    """
    self.build(inputs.shape)

    if not time_major:
      inputs = transpose(inputs, [1, 0, 2])

    result, state = self.fwd_gru(inputs, sequence_length, training)

    if self.bwd_gru is not None:
      inputs = reverse_sequence(inputs, sequence_length)
      bwd_result, bwd_state = self.bwd_gru(inputs, sequence_length, training)
      result = result, reverse_sequence(bwd_result, sequence_length)
      state = state, bwd_state

    if not time_major:
      result = transpose(result, [1, 0, 2])

    return result, state

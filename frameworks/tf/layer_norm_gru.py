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

"""Layer Normalized Gated Recurrent Unit"""


import pkg_resources
import tensorflow as tf

from tensorflow.compat import v1
from .base_rnn import BaseRNN
from .weight_config import WeightConfig


__all__ = [
    'LayerNormGRU'
]


LIB = tf.load_op_library(pkg_resources.resource_filename(__name__, 'libhaste_tf.so'))


@tf.RegisterGradient("HasteLayerNormGru")
def layer_norm_gru_gradient(op, *grads):
  training = op.get_attr('training')
  if not training:
    raise ValueError(('LayerNormGRU can only compute gradients if `training=True` was specified '
                      'during the forward pass.\nFailed op: {}').format(op.name))

  # Extract inputs and outputs from the op.
  x = op.inputs[0]
  W = op.inputs[1]
  R = op.inputs[2]
  bx = op.inputs[3]
  br = op.inputs[4]
  gamma = op.inputs[5]
  zoneout_mask = op.inputs[6]
  h = op.outputs[0]
  cache = op.outputs[1]

  # Pre-transpose matrices for better performance.
  x = tf.transpose(x, [2, 0, 1])
  W = tf.transpose(W, [1, 0])
  R = tf.transpose(R, [1, 0])

  grads = LIB.haste_layer_norm_gru_grad(x, W, R, bx, br, gamma, h, cache, grads[0], zoneout_mask)

  return [*grads, None]


class LayerNormGRULayer(tf.Module):
  def __init__(self,
        num_units,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        recurrent_bias_initializer=None,
        kernel_transform=None,
        recurrent_transform=None,
        bias_transform=None,
        recurrent_bias_transform=None,
        dropout=0.0,
        zoneout=0.0,
        dtype=None,
        name=None):
    super(LayerNormGRULayer, self).__init__(name)
    self.realname = name
    self.num_units = num_units

    identity = lambda x: x
    self.kernel_config = WeightConfig(v1.initializers.glorot_uniform(), None, identity)
    self.recurrent_config = WeightConfig(v1.initializers.orthogonal(), None, identity)
    self.bias_config = WeightConfig(v1.initializers.zeros(), None, identity)
    self.recurrent_bias_config = WeightConfig(v1.initializers.zeros(), None, identity)

    self.kernel_config.override(kernel_initializer, None, kernel_transform)
    self.recurrent_config.override(recurrent_initializer, None, recurrent_transform)
    self.bias_config.override(bias_initializer, None, bias_transform)
    self.recurrent_bias_config.override(recurrent_bias_initializer, None, recurrent_bias_transform)

    self.dropout = dropout
    self.zoneout = zoneout
    self.dtype = dtype or tf.float32
    self.built = False

  def build(self, shape):
    if self.built:
      return

    num_units = self.num_units
    input_size = int(shape[-1])

    def build_weights(initializer, shape):
      weights = [initializer(shape, dtype=self.dtype) for _ in range(3)]
      weights = tf.concat(weights, axis=-1)
      return weights

    kernel_weights = build_weights(self.kernel_config.initializer, [input_size, num_units])
    recurrent_weights = build_weights(self.recurrent_config.initializer, [num_units, num_units])
    biases = build_weights(self.bias_config.initializer, [num_units])
    recurrent_biases = build_weights(self.recurrent_bias_config.initializer, [num_units])

    weights = tf.concat([kernel_weights, recurrent_weights], axis=0)
    biases = tf.concat([biases, recurrent_biases], axis=0)

    with self.name_scope, v1.variable_scope(self.realname, 'gru_cell'):
      self._kernel = v1.get_variable('kernel', initializer=weights)
      self._bias = v1.get_variable('bias', initializer=biases)
      self.gamma = v1.get_variable('gamma', shape=[2, self.num_units * 3], initializer=v1.initializers.ones())
    self.built = True

  def get_weights(self):
    input_size = self._kernel.shape.as_list()[0] - self.num_units
    kernel, recurrent_kernel = tf.split(self._kernel, [input_size, self.num_units], axis=0)
    bias, recurrent_bias = tf.split(self._bias, 2, axis=0)
    return {
        'kernel': self.kernel_config.transform(kernel),
        'recurrent_kernel': self.recurrent_config.transform(recurrent_kernel),
        'bias': self.bias_config.transform(bias),
        'recurrent_bias': self.recurrent_bias_config.transform(recurrent_bias),
        'gamma': self.gamma,
    }

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
      zoneout_mask += tf.random.uniform([time_steps, batch_size, self.num_units], dtype=self.dtype)
      zoneout_mask = tf.floor(zoneout_mask)

    weights = self.get_weights()
    result, _ = LIB.haste_layer_norm_gru(
        inputs,
        weights['kernel'],
        tf.nn.dropout(weights['recurrent_kernel'], rate=self.dropout),
        weights['bias'],
        weights['recurrent_bias'],
        weights['gamma'],
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


class LayerNormGRU(BaseRNN):
  """
  Layer Normalized Gated Recurrent Unit layer.

  This GRU layer applies layer normalization to the input and recurrent output
  activations of a standard GRU. The implementation is fused and
  GPU-accelerated. There are two commonly-used variants of GRU cells. This one
  implements 1406.1078v1 which applies the reset gate to the hidden state
  after matrix multiplication. The other variant, 1406.1078v3, applies the
  reset gate before matrix multiplication and is currently unsupported.

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
      bias_initializer: (optional) the initializer to use for input bias
        vectors. Defaults to `zeros`.
      recurrent_bias_initializer: (optional) the initializer to use for
        recurrent bias vectors. Defaults to `zeros`.
      kernel_transform: (optional) a function with signature
        `(kernel: Tensor) -> Tensor` that transforms the kernel before it is
        used. Defaults to the identity function.
      recurrent_transform: (optional) a function with signature
        `(recurrent_kernel: Tensor) -> Tensor` that transforms the recurrent
        kernel before it is used. Defaults to the identity function.
      bias_transform: (optional) a function with signature
        `(bias: Tensor) -> Tensor` that transforms the bias before it is used.
        Defaults to the identity function.
      recurrent_bias_transform: (optional) a function with signature
        `(recurrent_bias: Tensor) -> Tensor` that transforms the recurrent bias
        before it is used. Defaults to the identity function.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix. Defaults to 0.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization. Defaults to 0.
      dtype: (optional) the data type for this layer. Defaults to `tf.float32`.
      name: (optional) string, the name for this layer.
    """
    super().__init__(LayerNormGRULayer, num_units, direction, 'gru_cell', **kwargs)

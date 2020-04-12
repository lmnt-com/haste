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

"""Independently Recurrent Neural Network"""


import pkg_resources
import tensorflow as tf

from tensorflow.compat import v1
from tensorflow.compat.v1.nn import rnn_cell

from .base_rnn import BaseRNN
from .weight_config import WeightConfig


__all__ = [
    'IndRNN'
]


LIB = tf.load_op_library(pkg_resources.resource_filename(__name__, 'libhaste_tf.so'))


@tf.RegisterGradient("HasteIndrnn")
def indrnn_gradient(op, *grads):
  training = op.get_attr('training')
  if not training:
    raise ValueError(('IndRNN can only compute gradients if `training=True` was specified during '
                      'the forward pass.\nFailed op: {}').format(op.name))

  # Extract inputs and outputs from the op.
  x = op.inputs[0]
  W = op.inputs[1]
  u = op.inputs[2]
  b = op.inputs[3]
  zoneout_mask = op.inputs[4]
  h = op.outputs[0]

  # Pre-transpose matrices for better performance.
  x = tf.transpose(x, [2, 0, 1])
  W = tf.transpose(W, [1, 0])

  dx, dW, du, db = LIB.haste_indrnn_grad(x, W, u, b, zoneout_mask, h, grads[0])
  return [dx, dW, du, db, None]


def _get_initializer(initializer):
  if not isinstance(initializer, dict):
    return initializer
  if 'uniform' in initializer:
    value = initializer['uniform']
    return v1.initializers.random_uniform(-value, value)
  if 'normal' in initializer:
    value = initializer['normal']
    return v1.initializers.truncated_normal(stddev=value)
  raise ValueError(f'Unknown initializer {initializer}')


class IndRNNLayer(tf.Module):
  def __init__(self,
        num_units,
        kernel_initializer=None,
        recurrent_initializer=None,
        bias_initializer=None,
        kernel_transform=None,
        recurrent_transform=None,
        bias_transform=None,
        zoneout=0.0,
        dtype=None,
        name=None):
    super().__init__(name)
    self.realname = name
    self.num_units = num_units

    identity = lambda x: x
    self.kernel_config = WeightConfig(v1.initializers.glorot_uniform(), None, identity)
    self.recurrent_config = WeightConfig(v1.initializers.random_uniform(-0.5, 0.5), None, identity)
    self.bias_config = WeightConfig(v1.initializers.zeros(), None, identity)

    self.kernel_config.override(_get_initializer(kernel_initializer), None, kernel_transform)
    self.recurrent_config.override(_get_initializer(recurrent_initializer), None, recurrent_transform)
    self.bias_config.override(_get_initializer(bias_initializer), None, bias_transform)

    self.zoneout = zoneout
    self.dtype = dtype or tf.float32
    self.kernel = None
    self.recurrent_scale = None
    self.bias = None
    self.recurrent_bias = None
    self.built = False

  def build(self, shape):
    if self.built:
      return

    num_units = self.num_units
    input_size = int(shape[-1])

    kernel_shape = tf.TensorShape([input_size, num_units])
    recurrent_shape = tf.TensorShape([num_units])
    bias_shape = tf.TensorShape([num_units])

    kernel_weights = self.kernel_config.initializer(kernel_shape, dtype=self.dtype)
    recurrent_weights = self.recurrent_config.initializer(recurrent_shape, dtype=self.dtype)
    biases = self.bias_config.initializer(bias_shape)

    with self.name_scope, v1.variable_scope(self.realname, 'indrnn_cell'):
      self.kernel = v1.get_variable('kernel', initializer=kernel_weights)
      self.recurrent_scale = v1.get_variable('recurrent_scale', initializer=recurrent_weights)
      self.bias = v1.get_variable('bias', initializer=biases)

    self.built = True

  def get_weights(self):
    return {
        'kernel': self.kernel_config.transform(self.kernel),
        'recurrent_scale': self.recurrent_config.transform(self.recurrent_scale),
        'bias': self.bias_config.transform(self.bias)
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
    result = LIB.haste_indrnn(
        inputs,
        weights['kernel'],
        weights['recurrent_scale'],
        weights['bias'],
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


class IndRNN(BaseRNN):
  """
  Independently Recurrent Neural Network layer.

  This layer offers a fused, GPU-accelerated TensorFlow op for inference and
  training. It also supports Zoneout regularization.
  """

  def __init__(self, num_units, direction='unidirectional', **kwargs):
    """
    Initialize the parameters of the IndRNN layer.

    Arguments:
      num_units: int, the number of units in the IndRNN cell.
      direction: string, 'unidirectional' or 'bidirectional'.
      **kwargs: Dict, keyword arguments (see below).

    Keyword Arguments:
      kernel_initializer: (optional) the initializer to use for the input
        matrix weights. Defaults to `glorot_uniform`.
      recurrent_initializer: (optional) the initializer to use for the
        recurrent scale weights. Defaults to uniform random in [-0.5, 0.5].
        Note that this initialization scheme is different than in the original
        authors' implementation. See https://github.com/lmnt-com/haste/issues/7
        for details.
      bias_initializer: (optional) the initializer to use for the bias vector.
        Defaults to `zeros`.
      kernel_transform: (optional) a function with signature
        `(kernel: Tensor) -> Tensor` that transforms the kernel before it is
        used. Defaults to the identity function.
      recurrent_transform: (optional) a function with signature
        `(recurrent_scale: Tensor) -> Tensor` that transforms the recurrent
        scale vector before it is used. Defaults to the identity function.
      bias_transform: (optional) a function with signature
        `(bias: Tensor) -> Tensor` that transforms the bias before it is used.
        Defaults to the identity function.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization. Defaults to 0.
      dtype: (optional) the data type for this layer. Defaults to `tf.float32`.
      name: (optional) string, the name for this layer.
    """
    super().__init__(IndRNNLayer, num_units, direction, 'indrnn_cell', **kwargs)

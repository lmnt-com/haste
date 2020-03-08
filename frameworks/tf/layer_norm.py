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

"""Layer Normalization"""


import pkg_resources
import tensorflow as tf

from tensorflow.compat import v1


__all__ = [
    'LayerNorm'
]


LIB = tf.load_op_library(pkg_resources.resource_filename(__name__, 'libhaste_tf.so'))


@tf.RegisterGradient("HasteLayerNorm")
def layer_norm_gradient(op, *grads):
  x = op.inputs[0]
  gamma = op.inputs[1]
  beta = op.inputs[2]
  cache = op.outputs[1]

  return LIB.haste_layer_norm_grad(x, gamma, beta, grads[0], cache)


class LayerNorm(tf.Module):
  """
  Layer normalization layer.

  This class exposes a fused and GPU-accelerated implementation of layer
  normalization as described by [Ba et al.](https://arxiv.org/abs/1607.06450)
  """

  def __init__(self, name=None):
    """
    Initialize the parameters of the layer normalization layer.

    Arguments:
      name: (optional) string, the name for this layer.
    """
    super(LayerNorm, self).__init__(name)
    self.realname = name
    self.gamma = None
    self.beta = None
    self.built = False

  def build(self, shape):
    """
    Creates the variables of the layer.

    Calling this method is optional for users of the LayerNorm class. It is
    called internally with the correct shape when `__call__` is invoked.

    Arguments:
      shape: instance of `TensorShape`.
    """
    if self.built:
      return
    hidden_size = int(shape[-1])
    with self.name_scope, v1.variable_scope(self.realname, 'layer_norm'):
      self.gamma = v1.get_variable('gamma', shape=[hidden_size], initializer=v1.initializers.ones())
      self.beta = v1.get_variable('beta', shape=[hidden_size], initializer=v1.initializers.zeros())
    self.built = True

  def __call__(self, x):
    """
    Runs the layer.

    Arguments:
      x: Tensor, a rank R tensor.

    Returns:
      y: Tensor, a rank R tensor with the last dimension normalized.
    """
    self.build(x.shape)
    y, _ = LIB.haste_layer_norm(x, self.gamma, self.beta)
    return y

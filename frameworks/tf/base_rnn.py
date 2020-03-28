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

"""Base RNN layer class."""


import tensorflow as tf

from tensorflow.compat import v1


__all__ = [
    'BaseRNN'
]


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


class BaseRNN(tf.Module):
  def __init__(self, rnn_class, num_units, direction, default_name, **kwargs):
    assert direction in ['unidirectional', 'bidirectional']

    self.default_name = default_name
    if direction == 'bidirectional':
      name = kwargs.pop('name', None)
      super().__init__(name)
      self.realname = name
      self.fw_layer = rnn_class(num_units, name='fw', **kwargs)
      self.bw_layer = rnn_class(num_units, name='bw', **kwargs)
    else:
      super().__init__()
      self.fw_layer = rnn_class(num_units, **kwargs)
      self.bw_layer = None

  def build(self, shape):
    """
    Creates the variables of the layer.

    Calling this method is optional for users of the RNN class. It is called
    internally with the correct shape when `__call__` is invoked.

    Arguments:
      shape: instance of `TensorShape`.
    """
    if self.bidirectional:
      with self.name_scope, v1.variable_scope(self.realname, self.default_name):
        self.fw_layer.build(shape)
        self.bw_layer.build(shape)
    else:
      self.fw_layer.build(shape)

  @property
  def output_size(self):
    if self.bidirectional:
      return self.fw_layer.output_size, self.bw_layer.output_size
    return self.fw_layer.output_size

  @property
  def state_size(self):
    if self.bidirectional:
      return self.fw_layer.state_size, self.bw_layer.state_size
    return self.fw_layer.state_size

  def __call__(self, inputs, training, sequence_length=None, time_major=False):
    """
    Runs the RNN layer.

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
      `([output_fw, output_bw], [state_fw, state_bw])` for bidirectional
      layers.
    """
    if not time_major:
      inputs = transpose(inputs, [1, 0, 2])

    result, state = self.fw_layer(inputs, sequence_length, training)

    if self.bidirectional:
      inputs = reverse_sequence(inputs, sequence_length)
      bw_result, bw_state = self.bw_layer(inputs, sequence_length, training)
      result = result, reverse_sequence(bw_result, sequence_length)
      state = state, bw_state

    if not time_major:
      result = transpose(result, [1, 0, 2])

    return result, state

  @property
  def bidirectional(self):
    """`True` if this is a bidirectional RNN, `False` otherwise."""
    return self.bw_layer is not None

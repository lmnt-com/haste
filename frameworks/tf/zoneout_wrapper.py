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

"""An RNN cell wrapper that applies Zoneout."""


import tensorflow as tf

from tensorflow.compat.v1.nn import rnn_cell


__all__ = [
    'ZoneoutWrapper'
]


class ZoneoutWrapper(rnn_cell.RNNCell):
  """
  An LSTM/GRU cell wrapper that applies zoneout to the inner cell's hidden state.

  The zoneout paper applies zoneout to both the cell state and hidden state,
  each with its own zoneout rate. This class (and the `LSTM` implementation in Haste)
  applies zoneout to the hidden state and not the cell state.
  """

  def __init__(self, cell, rate, training):
    """
    Initialize the parameters of the zoneout wrapper.

    Arguments:
      cell: RNNCell, an instance of {`BasicLSTMCell`, `LSTMCell`,
        `LSTMBlockCell`, `haste_tf.GRUCell`} on which to apply zoneout.
      rate: float, 0 <= rate <= 1, the percent of hidden units to zone out per
        time step.
      training: bool, `True` if used during training, `False` if used during
        inference.
    """
    super(ZoneoutWrapper, self).__init__()
    self.cell = cell
    self.rate = rate
    self.training = training

  @property
  def state_size(self):
    return self.cell.state_size

  @property
  def output_size(self):
    return self.cell.output_size

  def __call__(self, inputs, state, scope=None):
    """
    Runs one step of the RNN cell with zoneout applied.

    Arguments:
      see documentation for the inner cell.
    """

    output, new_state = self.cell(inputs, state, scope)

    # Zoneout disabled
    if not self.rate:
      return output, new_state

    if isinstance(new_state, rnn_cell.LSTMStateTuple):
      zoned_out_h = self._apply_zoneout(new_state.h, state.h)
      return output, rnn_cell.LSTMStateTuple(new_state.c, zoned_out_h)
    elif isinstance(new_state, list) and len(new_state) == 1:
      return output, self._apply_zoneout(new_state[0], state[0])
    elif isinstance(new_state, tf.Tensor):
      return output, self._apply_zoneout(new_state, state)
    else:
      raise ValueError(('ZoneoutWrapper wraps cells that return LSTMStateTuple or '
          'unnested state Tensors. Please use one of the following cell types:\n'
          '  tf.nn.rnn_cell.BasicLSTMCell\n'
          '  tf.nn.rnn_cell.LSTMCell\n'
          '  tf.contrib.rnn.LSTMBlockCell\n'
          '  haste_tf.GRUCell'))

  def _apply_zoneout(self, new_tensor, old_tensor):
    if self.training:
      mask = self._build_mask(tf.shape(new_tensor))
      zoned_out = (new_tensor - old_tensor) * mask + old_tensor
    else:
      zoned_out = self.rate * old_tensor + (1.0 - self.rate) * new_tensor
    return zoned_out

  def _build_mask(self, shape):
    mask = 1 - self.rate
    mask += tf.random.uniform(shape)
    return tf.floor(mask)

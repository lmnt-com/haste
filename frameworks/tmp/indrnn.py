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


import haste_pytorch_lib as LIB
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_rnn import BaseRNN


__all__ = [
    'IndRNN'
]


#@torch.jit.script
def IndRNNScript(
    training: bool,
    zoneout_prob: float,
    input,
    h0,
    kernel,
    recurrent_scale,
    bias,
    zoneout_mask):
  time_steps = input.shape[0]

  h = [h0]
  Wx = input @ kernel + bias
  for t in range(time_steps):
    h.append(torch.tanh(Wx[t] + h[-1] * recurrent_scale))
    if zoneout_prob:
      if training:
        h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
      else:
        h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]
  h = torch.stack(h)
  return h


class IndRNNFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, training, zoneout_prob, *inputs):
    h = LIB.indrnn_forward(training, zoneout_prob, *inputs)
    ctx.save_for_backward(inputs[0], *inputs[2:], h)
    ctx.training = training
    return h

  @staticmethod
  def backward(ctx, grad_h):
    if not ctx.training:
      raise RuntimeError('IndRNN backward can only be called in training mode')
    saved = [*ctx.saved_tensors]
    saved[0] = saved[0].permute(2, 0, 1).contiguous()
    saved[1] = saved[1].permute(1, 0).contiguous()
    grads = LIB.indrnn_backward(*saved, grad_h.contiguous())
    return (None, None, *grads, None)


class IndRNN(BaseRNN):
  """
  Independently Recurrent Neural Network layer.

  This layer offers a fused, GPU-accelerated PyTorch op for inference and
  training. It also supports Zoneout regularization.

  See [\_\_init\_\_](#__init__) and [forward](#forward) for usage.
  """

  def __init__(
      self,
      input_size,
      hidden_size,
      batch_first=False,
      zoneout=0.0,
      return_state_sequence=False):
    """
    Initialize the parameters of the IndRNN layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.
      return_state_sequence: (optional) bool, if `True`, the forward pass will
        return the entire state sequence instead of just the final state. Note
        that if the input is a padded sequence, the returned state will also
        be a padded sequence.

    Variables:
      kernel: the input projection weight matrix. Dimensions
        (input_size, hidden_size). Initialized with Xavier uniform
        initialization.
      recurrent_scale: the recurrent scale weight vector. Dimensions
        (hidden_size). Initialized uniformly in [-0.5, 0.5]. Note that this
        initialization scheme is different than in the original authors'
        implementation. See https://github.com/lmnt-com/haste/issues/7 for
        details.
      bias: the RNN bias vector. Dimensions (hidden_size). Initialized to zeros.
    """
    super().__init__(input_size, hidden_size, batch_first, zoneout, return_state_sequence)

    if zoneout < 0 or zoneout > 1:
      raise ValueError('IndRNN: zoneout must be in [0.0, 1.0]')

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_first = batch_first
    self.zoneout = zoneout

    self.kernel = nn.Parameter(torch.empty(input_size, hidden_size))
    self.recurrent_scale = nn.Parameter(torch.empty(hidden_size))
    self.bias = nn.Parameter(torch.empty(hidden_size))
    self.reset_parameters()

  def reset_parameters(self):
    nn.init.xavier_uniform_(self.kernel)
    nn.init.uniform_(self.recurrent_scale, -0.5, 0.5)
    nn.init.zeros_(self.bias)

  def forward(self, input, state=None, lengths=None):
    """
    Runs a forward pass of the IndRNN layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the GRU.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      state: (optional) Tensor, the initial state for each batch element in
        `input`. Dimensions (1, batch_size, hidden_size). Defaults to zeros.
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size). This argument may be omitted if
        all batch elements are unpadded and have the same sequence length.

    Returns:
      output: Tensor, the output of the GRU layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
        that if `lengths` was specified, the `output` tensor will not be
        masked. It's the caller's responsibility to either not use the invalid
        entries or to mask them out before using them.
      state: the hidden state for the last sequence item. Dimensions
        (1, batch_size, hidden_size).
    """
    input = self._permute(input)
    state_shape = [1, input.shape[1], self.hidden_size]
    h0 = self._get_state(input, state, state_shape)
    h = self._impl(input, h0[0], self._get_zoneout_mask(input))
    state = self._get_final_state(h, lengths)
    output = self._permute(h[1:])
    return output, state

  def _impl(self, input, state, zoneout_mask):
    if self._is_cuda():
      return IndRNNFunction.apply(
        self.training,
        self.zoneout,
        input.contiguous(),
        state.contiguous(),
        self.kernel.contiguous(),
        self.recurrent_scale.contiguous(),
        self.bias.contiguous(),
        zoneout_mask.contiguous())
    else:
      return IndRNNScript(
        self.training,
        self.zoneout,
        input.contiguous(),
        state.contiguous(),
        self.kernel.contiguous(),
        self.recurrent_scale.contiguous(),
        self.bias.contiguous(),
        zoneout_mask.contiguous())

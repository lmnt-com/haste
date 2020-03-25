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
  def __init__(
      self,
      input_size,
      hidden_size,
      batch_first=False,
      zoneout=0.0):
    super().__init__(input_size, hidden_size, batch_first, zoneout)

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
    nn.init.uniform_(self.recurrent_scale, -1.0, 1.0)
    nn.init.zeros_(self.bias)

  def forward(self, input, state=None, lengths=None):
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

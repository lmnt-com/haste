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


__all__ = [
    'LSTM'
]


class LSTMFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, training, zoneout_prob, *inputs):
    h, c, cache = LIB.lstm_forward(training, zoneout_prob, *inputs)
    ctx.save_for_backward(*inputs, h, c, cache)
    ctx.mark_non_differentiable(inputs[-1])
    ctx.training = training
    return h, c

  @staticmethod
  def backward(ctx, grad_h, grad_c):
    if not ctx.training:
      raise RuntimeError('LSTM backward can only be called in training mode')

    saved = [*ctx.saved_tensors]
    saved[0] = saved[0].permute(2, 0, 1).contiguous()
    saved[1] = saved[1].permute(1, 0).contiguous()
    saved[2] = saved[2].permute(1, 0).contiguous()
    grads = LIB.lstm_backward(*saved, grad_h.contiguous(), grad_c.contiguous())
    return (None, None, *grads, None)

class LSTM(nn.Module):
  def __init__(self,
      input_size,
      hidden_size,
      batch_first=False,
      forget_bias=1.0,
      dropout=0.0,
      zoneout=0.0):
    super(LSTM, self).__init__()

    if dropout < 0 or dropout > 1:
      raise ValueError('LSTM: dropout must be in [0.0, 1.0]')
    if zoneout < 0 or zoneout > 1:
      raise ValueError('LSTM: zoneout must be in [0.0, 1.0]')

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_first = batch_first
    self.forget_bias = forget_bias
    self.dropout = dropout
    self.zoneout = zoneout

    gpu = torch.device('cuda')
    self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 4, device=gpu))
    self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 4, device=gpu))
    self.bias = nn.Parameter(torch.empty(hidden_size * 4, device=gpu))
    self.reset_parameters()

  def reset_parameters(self):
    hidden_size = self.hidden_size
    for i in range(4):
      nn.init.xavier_uniform_(self.kernel[:, i*hidden_size:(i+1)*hidden_size])
      nn.init.orthogonal_(self.recurrent_kernel[:, i*hidden_size:(i+1)*hidden_size])
    nn.init.zeros_(self.bias)
    nn.init.constant_(self.bias[hidden_size*2:hidden_size*3], self.forget_bias)

  def forward(self, input):
    if self.batch_first:
      input = input.permute(1, 0, 2)

    if self.zoneout:
      zoneout_mask = torch.empty(
          input.shape[0],
          input.shape[1],
          self.hidden_size,
          dtype=input.dtype,
          device=input.device)
      zoneout_mask.bernoulli_(1.0 - self.zoneout)
    else:
      zoneout_mask = torch.empty(0, dtype=input.dtype, device=input.device)
    h, c = LSTMFunction.apply(
        self.training,
        self.zoneout,
        input.contiguous(),
        self.kernel.contiguous(),
        F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
        self.bias.contiguous(),
        zoneout_mask.contiguous())

    output = h[1:]
    state = (h[-1].unsqueeze(0), c[-1].unsqueeze(0))

    if self.batch_first:
      output = output.permute(1, 0, 2)

    return output, state

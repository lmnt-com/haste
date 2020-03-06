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
    'GRU'
]


class GRUFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, training, zoneout_prob, *inputs):
    h, cache = LIB.gru_forward(training, zoneout_prob, *inputs)
    ctx.save_for_backward(*inputs, h, cache)
    ctx.mark_non_differentiable(inputs[-1])
    ctx.training = training
    return h

  @staticmethod
  def backward(ctx, grad_h):
    if not ctx.training:
      raise RuntimeError('GRU backward can only be called in training mode')

    saved = [*ctx.saved_tensors]
    saved[0] = saved[0].permute(0, 2, 1).contiguous()
    saved[1] = saved[1].permute(1, 0).contiguous()
    saved[2] = saved[2].permute(1, 0).contiguous()
    saved[-2] = saved[-2].permute(0, 2, 1).contiguous()
    grads = LIB.gru_backward(*saved, grad_h.contiguous())
    return (None, None, *grads, None)


class GRU(nn.Module):
  def __init__(self,
      input_size,
      hidden_size,
      batch_first=False,
      dropout=0.0,
      zoneout=0.0):
    super(GRU, self).__init__()

    if zoneout < 0 or zoneout > 1:
      raise ValueError('GRU: zoneout must be in [0.0, 1.0]')

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_first = batch_first
    self.dropout = dropout
    self.zoneout = zoneout

    gpu = torch.device('cuda')
    self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 3, device=gpu))
    self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3, device=gpu))
    self.bias = nn.Parameter(torch.empty(hidden_size * 3, device=gpu))
    self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3, device=gpu))
    self.reset_parameters()

  def reset_parameters(self):
    hidden_size = self.hidden_size
    for i in range(3):
      nn.init.xavier_uniform_(self.kernel[:, i*hidden_size:(i+1)*hidden_size])
      nn.init.orthogonal_(self.recurrent_kernel[:, i*hidden_size:(i+1)*hidden_size])
    nn.init.zeros_(self.bias)
    nn.init.zeros_(self.recurrent_bias)

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
      zoneout_mask = torch.empty(0, 0, 0, dtype=input.dtype, device=input.device)
    h = GRUFunction.apply(
        self.training,
        self.zoneout,
        input.contiguous(),
        self.kernel.contiguous(),
        F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
        self.bias.contiguous(),
        self.recurrent_bias.contiguous(),
        zoneout_mask.contiguous())

    output = h
    state = h[-1].unsqueeze(0)

    if self.batch_first:
      output = output.permute(1, 0, 2)

    return output, state

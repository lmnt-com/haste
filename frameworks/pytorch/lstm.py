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

"""Long Short-Term Memory"""


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
  """
  Long Short-Term Memory layer.

  This LSTM layer offers a fused, GPU-accelerated PyTorch op for inference
  and training. Although this implementation is comparable in performance to
  cuDNN's LSTM, it offers additional options not typically found in other
  high-performance implementations. DropConnect and Zoneout regularization are
  built-in, and this layer allows setting a non-zero initial forget gate bias.

  See [\_\_init\_\_](#__init__) and [forward](#forward) for general usage.
  See [from_native_weights](#from_native_weights) and
  [to_native_weights](#to_native_weights) for compatibility with PyTorch LSTMs.
  """

  def __init__(self,
      input_size,
      hidden_size,
      batch_first=False,
      forget_bias=1.0,
      dropout=0.0,
      zoneout=0.0):
    """
    Initialize the parameters of the LSTM layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      forget_bias: (optional) float, sets the initial bias of the forget gate
        for this LSTM cell.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.

    Variables:
      kernel: the input projection weight matrix. Dimensions
        (input_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
        with Xavier uniform initialization.
      recurrent_kernel: the recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 4) with `i,g,f,o` gate layout. Initialized
        with orthogonal initialization.
      bias: the projection bias vector. Dimensions (hidden_size * 4) with
        `i,g,f,o` gate layout. The forget gate biases are initialized to
        `forget_bias` and the rest are zeros.
    """
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

  def to_native_weights(self):
    """
    Converts Haste LSTM weights to native PyTorch LSTM weights.

    Returns:
      weight_ih_l0: Parameter, the input-hidden weights of the LSTM layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the LSTM layer.
      bias_ih_l0: Parameter, the input-hidden bias of the LSTM layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the LSTM layer.
    """
    def reorder_weights(w):
      i, g, f, o = torch.chunk(w, 4, dim=-1)
      return torch.cat([i, f, g, o], dim=-1)
    kernel = reorder_weights(self.kernel).permute(1, 0).contiguous()
    recurrent_kernel = reorder_weights(self.recurrent_kernel).permute(1, 0).contiguous()
    half_bias = reorder_weights(self.bias) / 2.0

    kernel = torch.nn.Parameter(kernel)
    recurrent_kernel = torch.nn.Parameter(recurrent_kernel)
    bias1 = torch.nn.Parameter(half_bias)
    bias2 = torch.nn.Parameter(half_bias.clone())
    return kernel, recurrent_kernel, bias1, bias2

  def from_native_weights(self, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0):
    """
    Copies and converts the provided PyTorch LSTM weights into this layer.

    Arguments:
      weight_ih_l0: Parameter, the input-hidden weights of the PyTorch LSTM layer.
      weight_hh_l0: Parameter, the hidden-hidden weights of the PyTorch LSTM layer.
      bias_ih_l0: Parameter, the input-hidden bias of the PyTorch LSTM layer.
      bias_hh_l0: Parameter, the hidden-hidden bias of the PyTorch LSTM layer.
    """
    def reorder_weights(w):
      i, f, g, o = torch.chunk(w, 4, dim=-1)
      return torch.cat([i, g, f, o], dim=-1)
    kernel = reorder_weights(weight_ih_l0.permute(1, 0)).contiguous().cuda()
    recurrent_kernel = reorder_weights(weight_hh_l0.permute(1, 0)).contiguous().cuda()
    bias = reorder_weights(bias_ih_l0 + bias_hh_l0).contiguous().cuda()

    self.kernel = nn.Parameter(kernel)
    self.recurrent_kernel = nn.Parameter(recurrent_kernel)
    self.bias = nn.Parameter(bias)

  def reset_parameters(self):
    """Resets this layer's parameters to their initial values."""
    hidden_size = self.hidden_size
    for i in range(4):
      nn.init.xavier_uniform_(self.kernel[:, i*hidden_size:(i+1)*hidden_size])
      nn.init.orthogonal_(self.recurrent_kernel[:, i*hidden_size:(i+1)*hidden_size])
    nn.init.zeros_(self.bias)
    nn.init.constant_(self.bias[hidden_size*2:hidden_size*3], self.forget_bias)

  def forward(self, input, lengths=None):
    """
    Runs a forward pass of the LSTM layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the LSTM.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      lengths: (optional) Tensor, list of sequence lengths for each batch
        element. Dimension (batch_size). This argument may be omitted if
        all batch elements are unpadded and have the same sequence length.

    Returns:
      output: Tensor, the output of the LSTM layer. Dimensions
        (seq_len, batch_size, hidden_size) if `batch_first` is `False` (default)
        or (batch_size, seq_len, hidden_size) if `batch_first` is `True`. Note
        that if `lengths` was specified, the `output` tensor will not be
        masked. It's the caller's responsibility to either not use the invalid
        entries or to mask them out before using them.
      (h_n, c_n): the hidden and cell states, respectively, for the last
        sequence item. Dimensions (1, batch_size, hidden_size).
    """
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

    if lengths is not None:
      cols = range(h.size(1))
      state = (h[[lengths, cols]].unsqueeze(0), c[[lengths, cols]].unsqueeze(0))
    else:
      state = (h[-1].unsqueeze(0), c[-1].unsqueeze(0))

    output = h[1:]
    if self.batch_first:
      output = output.permute(1, 0, 2)

    return output, state

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


import haste_pytorch_lib as LIB
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_rnn import BaseRNN


__all__ = [
    'LayerNormGRU'
]


#@torch.jit.script
def LayerNormGRUScript(
    training: bool,
    zoneout_prob: float,
    input,
    h0,
    kernel,
    recurrent_kernel,
    bias,
    recurrent_bias,
    gamma,
    zoneout_mask):
  time_steps = input.shape[0]
  batch_size = input.shape[1]
  hidden_size = recurrent_kernel.shape[0]

  h = [h0]
  Wx = F.layer_norm(input @ kernel, (hidden_size * 3,), weight=gamma[0]) + bias
  for t in range(time_steps):
    Rh = F.layer_norm(h[t] @ recurrent_kernel, (hidden_size * 3,), weight=gamma[1]) + recurrent_bias
    vx = torch.chunk(Wx[t], 3, 1)
    vh = torch.chunk(Rh, 3, 1)

    z = torch.sigmoid(vx[0] + vh[0])
    r = torch.sigmoid(vx[1] + vh[1])
    g = torch.tanh(vx[2] + r * vh[2])

    h.append(z * h[t] + (1 - z) * g)
    if zoneout_prob:
      if training:
        h[-1] = (h[-1] - h[-2]) * zoneout_mask[t] + h[-2]
      else:
        h[-1] = zoneout_prob * h[-2] + (1 - zoneout_prob) * h[-1]

  h = torch.stack(h)
  return h


class LayerNormGRUFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, training, zoneout_prob, *inputs):
    output = LIB.layer_norm_gru_forward(training, zoneout_prob, *inputs)
    ctx.save_for_backward(inputs[0], *inputs[2:], *output)
    ctx.mark_non_differentiable(inputs[-1])
    ctx.training = training
    return output[0]

  @staticmethod
  def backward(ctx, grad_h):
    if not ctx.training:
      raise RuntimeError('LayerNormGRU backward can only be called in training mode')

    saved = [*ctx.saved_tensors]
    saved[0] = saved[0].permute(2, 0, 1).contiguous()
    saved[1] = saved[1].permute(1, 0).contiguous()
    saved[2] = saved[2].permute(1, 0).contiguous()
    grads = LIB.layer_norm_gru_backward(*saved, grad_h.contiguous())
    return (None, None, *grads, None)


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

  See [\_\_init\_\_](#__init__) and [forward](#forward) for usage.
  """

  def __init__(self,
      input_size,
      hidden_size,
      batch_first=False,
      dropout=0.0,
      zoneout=0.0):
    """
    Initialize the parameters of the GRU layer.

    Arguments:
      input_size: int, the feature dimension of the input.
      hidden_size: int, the feature dimension of the output.
      batch_first: (optional) bool, if `True`, then the input and output
        tensors are provided as `(batch, seq, feature)`.
      dropout: (optional) float, sets the dropout rate for DropConnect
        regularization on the recurrent matrix.
      zoneout: (optional) float, sets the zoneout rate for Zoneout
        regularization.

    Variables:
      kernel: the input projection weight matrix. Dimensions
        (input_size, hidden_size * 3) with `z,r,h` gate layout. Initialized
        with Xavier uniform initialization.
      recurrent_kernel: the recurrent projection weight matrix. Dimensions
        (hidden_size, hidden_size * 3) with `z,r,h` gate layout. Initialized
        with orthogonal initialization.
      bias: the input projection bias vector. Dimensions (hidden_size * 3) with
        `z,r,h` gate layout. Initialized to zeros.
      recurrent_bias: the recurrent projection bias vector. Dimensions
        (hidden_size * 3) with `z,r,h` gate layout. Initialized to zeros.
      gamma: the input and recurrent normalization gain. Dimensions
        (2, hidden_size * 4) with `gamma[0]` specifying the input gain and
        `gamma[1]` specifying the recurrent gain. Initialized to ones.
    """
    super().__init__(input_size, hidden_size, batch_first, zoneout)

    if dropout < 0 or dropout > 1:
      raise ValueError('LayerNormGRU: dropout must be in [0.0, 1.0]')
    if zoneout < 0 or zoneout > 1:
      raise ValueError('LayerNormGRU: zoneout must be in [0.0, 1.0]')

    self.dropout = dropout

    self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 3))
    self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3))
    self.bias = nn.Parameter(torch.empty(hidden_size * 3))
    self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))
    self.gamma = nn.Parameter(torch.empty(2, hidden_size * 3))
    self.reset_parameters()

  def reset_parameters(self):
    """Resets this layer's parameters to their initial values."""
    hidden_size = self.hidden_size
    for i in range(3):
      nn.init.xavier_uniform_(self.kernel[:, i*hidden_size:(i+1)*hidden_size])
      nn.init.orthogonal_(self.recurrent_kernel[:, i*hidden_size:(i+1)*hidden_size])
    nn.init.zeros_(self.bias)
    nn.init.zeros_(self.recurrent_bias)
    nn.init.ones_(self.gamma)

  def forward(self, input, state=None, lengths=None):
    """
    Runs a forward pass of the GRU layer.

    Arguments:
      input: Tensor, a batch of input sequences to pass through the GRU.
        Dimensions (seq_len, batch_size, input_size) if `batch_first` is
        `False`, otherwise (batch_size, seq_len, input_size).
      state: (optional) Tensor, the intial state for each batch element in
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
      h_n: the hidden state for the last sequence item. Dimensions
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
      return LayerNormGRUFunction.apply(
          self.training,
          self.zoneout,
          input.contiguous(),
          state.contiguous(),
          self.kernel.contiguous(),
          F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
          self.bias.contiguous(),
          self.recurrent_bias.contiguous(),
          self.gamma.contiguous(),
          zoneout_mask.contiguous())
    else:
      return LayerNormGRUScript(
          self.training,
          self.zoneout,
          input.contiguous(),
          state.contiguous(),
          self.kernel.contiguous(),
          F.dropout(self.recurrent_kernel, self.dropout, self.training).contiguous(),
          self.bias.contiguous(),
          self.recurrent_bias.contiguous(),
          self.gamma.contiguous(),
          zoneout_mask.contiguous())

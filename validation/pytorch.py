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

import argparse
from unittest import mock
import torch
import haste_pytorch as haste


RNN_MAP = {
    'gru': haste.GRU,
    'indrnn': haste.IndRNN,
    'layer_norm_gru': haste.LayerNormGRU,
    'layer_norm_indrnn': haste.LayerNormIndRNN,
    'layer_norm_lstm': haste.LayerNormLSTM,
    'lstm': haste.LSTM,
}

HASTE_TO_NATIVE = {
    haste.GRU: torch.nn.GRU,
    haste.LSTM: torch.nn.LSTM,
}

batch_size = 32
time_steps = 250
input_size = 128
hidden_size = 256


def self_consistency(rnn, x):
  x_cuda = x.clone().cuda()
  x_cuda_torch = x_cuda.detach().clone()
  x_cuda.requires_grad_(True)
  x_cuda_torch.requires_grad_(True)

  rnn.cuda()
  seed = 5566
  torch.manual_seed(seed)
  y1, _ = rnn.forward(x_cuda)
  y1.backward(torch.ones_like(y1))
  torch.manual_seed(seed)
  with mock.patch.object(rnn, "_is_cuda", lambda: False):
    y2, _ = rnn.forward(x_cuda_torch)
  y2.backward(torch.ones_like(y2))

  g1 = x_cuda_torch.grad.data
  g2 = x_cuda.grad.data

  print(torch.max(torch.abs(y1.cpu()-y2.cpu())))
  print(torch.max(torch.abs(g1.cpu()-g2.cpu())))


def native_consistency(haste_rnn, pytorch_rnn, x):
  pytorch_rnn.cuda()
  haste_rnn.cuda()
  haste_rnn.from_native_weights(
      pytorch_rnn.weight_ih_l0,
      pytorch_rnn.weight_hh_l0,
      pytorch_rnn.bias_ih_l0,
      pytorch_rnn.bias_hh_l0)

  x1 = x.clone().cuda()
  x2 = x.clone().cuda()
  x1.requires_grad_(True)
  x2.requires_grad_(True)

  y1, _ = haste_rnn.forward(x1)
  y1.backward(torch.ones_like(y1))

  y2, _ = pytorch_rnn.forward(x2)
  y2.backward(torch.ones_like(y2))

  g1 = x1.grad.data
  g2 = x2.grad.data

  print(torch.max(torch.abs(y1-y2)))
  print(torch.max(torch.abs(g1-g2)))


def _run_rnn(rnn_type, x, **kwargs):
  rnn = rnn_type(input_size, hidden_size, **kwargs)
  self_consistency(rnn, x)
  if rnn_type in HASTE_TO_NATIVE and not kwargs:
    pytorch_rnn = HASTE_TO_NATIVE[rnn_type](input_size, hidden_size)
    native_consistency(rnn, pytorch_rnn, x)


def run_rnn(rnn_type, x):
  for kwargs in [dict(), dict(zoneout=0.5)]:
    _run_rnn(rnn_type, x, **kwargs)


def main(args):
  x = torch.rand(time_steps, batch_size, input_size)
  if args.rnn_type == 'all':
    for type_name, rnn_type in RNN_MAP.items():
      print(f'[{type_name}]')
      run_rnn(rnn_type, x)
      print('')
  else:
    print(f'[{args.rnn_type}]')
    rnn_type = RNN_MAP[args.rnn_type]
    rnn = run_rnn(rnn_type, x)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'rnn_type',
      nargs='?',
      default='all',
      choices=list(RNN_MAP.keys()) + ['all'])
  main(parser.parse_args())

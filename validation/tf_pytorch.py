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
import numpy as np
import tensorflow as tf
import haste_tf
import torch
import torch.nn as nn
import haste_pytorch


def stfu():
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def copy_weights_gru(rnn_tf, rnn_pt):
  weights = rnn_tf.fw_layer.get_weights()
  kernel = torch.Tensor(weights['kernel'].numpy())
  recurrent_kernel = torch.Tensor(weights['recurrent_kernel'].numpy())
  bias = torch.Tensor(weights['bias'].numpy())
  recurrent_bias = torch.Tensor(weights['recurrent_bias'].numpy())

  rnn_pt.kernel = nn.Parameter(kernel)
  rnn_pt.recurrent_kernel = nn.Parameter(recurrent_kernel)
  rnn_pt.bias = nn.Parameter(bias)
  rnn_pt.recurrent_bias = nn.Parameter(recurrent_bias)


def copy_weights_indrnn(rnn_tf, rnn_pt):
  weights = rnn_tf.fw_layer.get_weights()
  kernel = torch.Tensor(weights['kernel'].numpy())
  recurrent_scale = torch.Tensor(weights['recurrent_scale'].numpy())
  bias = torch.Tensor(weights['bias'].numpy())

  rnn_pt.kernel = nn.Parameter(kernel)
  rnn_pt.recurrent_scale = nn.Parameter(recurrent_scale)
  rnn_pt.bias = nn.Parameter(bias)


def copy_weights_layer_norm_gru(rnn_tf, rnn_pt):
  weights = rnn_tf.fw_layer.get_weights()
  kernel = torch.Tensor(weights['kernel'].numpy())
  recurrent_kernel = torch.Tensor(weights['recurrent_kernel'].numpy())
  bias = torch.Tensor(weights['bias'].numpy())
  recurrent_bias = torch.Tensor(weights['recurrent_bias'].numpy())
  gamma = torch.Tensor(weights['gamma'].numpy())

  rnn_pt.kernel = nn.Parameter(kernel)
  rnn_pt.recurrent_kernel = nn.Parameter(recurrent_kernel)
  rnn_pt.bias = nn.Parameter(bias)
  rnn_pt.recurrent_bias = nn.Parameter(recurrent_bias)
  rnn_pt.gamma = nn.Parameter(gamma)


def copy_weights_layer_norm_lstm(rnn_tf, rnn_pt):
  weights = rnn_tf.fw_layer.get_weights()
  kernel = torch.Tensor(weights['kernel'].numpy())
  recurrent_kernel = torch.Tensor(weights['recurrent_kernel'].numpy())
  bias = torch.Tensor(weights['bias'].numpy())
  gamma = torch.Tensor(weights['gamma'].numpy())
  gamma_h = torch.Tensor(weights['gamma_h'].numpy())
  beta_h = torch.Tensor(weights['beta_h'].numpy())

  rnn_pt.kernel = nn.Parameter(kernel)
  rnn_pt.recurrent_kernel = nn.Parameter(recurrent_kernel)
  rnn_pt.bias = nn.Parameter(bias)
  rnn_pt.gamma = nn.Parameter(gamma)
  rnn_pt.gamma_h = nn.Parameter(gamma_h)
  rnn_pt.beta_h = nn.Parameter(beta_h)


def copy_weights_lstm(rnn_tf, rnn_pt):
  weights = rnn_tf.fw_layer.get_weights()
  kernel = torch.Tensor(weights['kernel'].numpy())
  recurrent_kernel = torch.Tensor(weights['recurrent_kernel'].numpy())
  bias = torch.Tensor(weights['bias'].numpy())

  rnn_pt.kernel = nn.Parameter(kernel)
  rnn_pt.recurrent_kernel = nn.Parameter(recurrent_kernel)
  rnn_pt.bias = nn.Parameter(bias)


batch_size = 32
time_steps = 250
input_size = 128
hidden_size = 256

RNN_MAP = {
    'gru': haste_tf.GRU,
    'indrnn': haste_tf.IndRNN,
    'layer_norm_gru': haste_tf.LayerNormGRU,
    'layer_norm_lstm': haste_tf.LayerNormLSTM,
    'lstm': haste_tf.LSTM,
}

TF_TO_PT = {
    haste_tf.GRU: haste_pytorch.GRU,
    haste_tf.IndRNN: haste_pytorch.IndRNN,
    haste_tf.LayerNormGRU: haste_pytorch.LayerNormGRU,
    haste_tf.LayerNormLSTM: haste_pytorch.LayerNormLSTM,
    haste_tf.LSTM: haste_pytorch.LSTM,
}

WEIGHT_COPY_MAP = {
    haste_tf.GRU: copy_weights_gru,
    haste_tf.IndRNN: copy_weights_indrnn,
    haste_tf.LayerNormGRU: copy_weights_layer_norm_gru,
    haste_tf.LayerNormLSTM: copy_weights_layer_norm_lstm,
    haste_tf.LSTM: copy_weights_lstm,
}


def run_rnn(rnn_type, x):
  rnn_tf = rnn_type(hidden_size)
  rnn_pt = TF_TO_PT[rnn_type](input_size, hidden_size, batch_first=True)

  rnn_tf.build(x.shape)
  WEIGHT_COPY_MAP[type(rnn_tf)](rnn_tf, rnn_pt)

  x1 = tf.convert_to_tensor(x)
  x2 = torch.Tensor(x)
  x2.requires_grad_(True)
  with tf.GradientTape() as tape:
    tape.watch(x1)
    y1, _ = rnn_tf(x1, training=True)
    g1 = tape.gradient(y1, x1)

  y2, _ = rnn_pt(x2)
  y2.backward(torch.ones_like(y2))

  print(np.amax(np.abs(y1.numpy() - y2.detach().numpy())))
  print(np.amax(np.abs(g1.numpy() - x2.grad.data.numpy())))


def main(args):
  tf.compat.v1.enable_eager_execution()
  stfu()

  x = np.random.normal(size=[time_steps, batch_size, input_size]).astype(np.float32)
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

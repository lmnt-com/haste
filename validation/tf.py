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
import haste_tf as haste
import tensorflow as tf


def stfu():
  import os
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def NativeGRUBuilder(hidden_size):
  return tf.keras.layers.GRU(
      hidden_size,
      implementation=2,
      activation='tanh',
      recurrent_activation='sigmoid',
      return_sequences=True,
      reset_after=True)


def NativeLSTMBuilder(hidden_size):
  return tf.keras.layers.LSTM(
      hidden_size,
      implementation=2,
      activation='tanh',
      unit_forget_bias=False,
      recurrent_activation='sigmoid',
      return_sequences=True)


def NativeGRUWeights(native_gru, haste_gru):
  weights = haste_gru.fw_layer.get_weights()
  native_gru.variables[0].assign(weights['kernel'])
  native_gru.variables[1].assign(weights['recurrent_kernel'])
  native_gru.variables[2].assign(tf.stack([weights['bias'], weights['recurrent_bias']], axis=0))


def NativeLSTMWeights(native_lstm, haste_lstm):
  def swapple(x):
    i, g, f, o = tf.split(x, 4, axis=-1)
    return tf.concat([i, f, g, o], axis=-1)
  weights = haste_lstm.fw_layer.get_weights()
  native_lstm.variables[0].assign(swapple(weights['kernel']))
  native_lstm.variables[1].assign(swapple(weights['recurrent_kernel']))
  native_lstm.variables[2].assign(swapple(weights['bias']))


RNN_MAP = {
    'gru': haste.GRU,
    'indrnn': haste.IndRNN,
    'layer_norm_gru': haste.LayerNormGRU,
    'layer_norm_lstm': haste.LayerNormLSTM,
    'lstm': haste.LSTM,
}

HASTE_TO_NATIVE = {
    haste.GRU: NativeGRUBuilder,
    haste.LSTM: NativeLSTMBuilder,
}

HASTE_TO_NATIVE_WEIGHTS = {
    haste.GRU: NativeGRUWeights,
    haste.LSTM: NativeLSTMWeights,
}


batch_size = 32
time_steps = 250
input_size = 128
hidden_size = 256


def native_consistency(haste_rnn, native_rnn, x):
  with tf.GradientTape() as tape:
    tape.watch(x)
    y1, _ = haste_rnn(x, training=True)
    g1 = tape.gradient(y1, x)

  native_rnn.build(x.shape)
  HASTE_TO_NATIVE_WEIGHTS[type(haste_rnn)](native_rnn, haste_rnn)

  with tf.GradientTape() as tape:
    tape.watch(x)
    y2 = native_rnn(x, training=True)
    g2 = tape.gradient(y2, x)

  print(tf.reduce_max(tf.abs(y2-y1)))
  print(tf.reduce_max(tf.abs(g2-g1)))


def run_rnn(rnn_type, x):
  rnn = rnn_type(hidden_size)
  if rnn_type in HASTE_TO_NATIVE:
    native_rnn = HASTE_TO_NATIVE[rnn_type](hidden_size)
    native_consistency(rnn, native_rnn, x)


def main(args):
  tf.compat.v1.enable_eager_execution()
  stfu()

  x = tf.random.normal([batch_size, time_steps, input_size])
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

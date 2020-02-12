# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def extract(x, predicate):
  return np.array(list(filter(predicate, x)))


def main(args):
  np.set_printoptions(suppress=True)

  A = np.loadtxt(args.A, delimiter=',')
  B = np.loadtxt(args.B, delimiter=',')

  faster = 1.0 - A[:,-1] / B[:,-1]

  print(f'A is faster than B by:')
  print(f'  mean:   {np.mean(faster)*100:7.4}%')
  print(f'  std:    {np.std(faster)*100:7.4}%')
  print(f'  median: {np.median(faster)*100:7.4}%')
  print(f'  min:    {np.min(faster)*100:7.4}%')
  print(f'  max:    {np.max(faster)*100:7.4}%')

  for batch_size in np.unique(A[:,0]):
    for input_size in np.unique(A[:,2]):
      a = extract(A, lambda x: x[0] == batch_size and x[2] == input_size)
      b = extract(B, lambda x: x[0] == batch_size and x[2] == input_size)
      fig, ax = plt.subplots(dpi=200)
      ax.set_xticks(a[:,1])
      ax.set_xticklabels(a[:,1].astype(np.int32), rotation=60)
      ax.tick_params(axis='y', which='both', length=0)
      ax.spines['top'].set_visible(False)
      ax.spines['right'].set_visible(False)
      plt.title(f'batch size={int(batch_size)}, input size={int(input_size)}')
      plt.plot(a[:,1], a[:,-1], color=args.color[0])
      plt.plot(a[:,1], b[:,-1], color=args.color[1])
      plt.xlabel('hidden size')
      plt.ylabel('time (ms)')
      plt.legend(args.name, frameon=False)
      plt.tight_layout()
      if args.save:
        os.makedirs(args.save[0], exist_ok=True)
        plt.savefig(f'{args.save[0]}/report_n={int(batch_size)}_c={int(input_size)}.png', dpi=200)
      else:
        plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', nargs=2, default=['A', 'B'])
  parser.add_argument('--color', nargs=2, default=['#1f77b4', '#2ca02c'])
  parser.add_argument('--save', nargs=1, default=None)
  parser.add_argument('A')
  parser.add_argument('B')
  main(parser.parse_args())

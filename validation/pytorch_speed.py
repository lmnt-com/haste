import torch
import haste_pytorch as haste

from time import time


seq_len = 2500
batch_size = 64
input_size = 256
hidden_size = 4096

rnn = haste.IndRNN(input_size, hidden_size).cuda()

x = torch.rand(seq_len, batch_size, input_size).cuda()
start = time()
for _ in range(10):
  y, _ = rnn(x)
  y.backward(torch.ones_like(y))
end = time()
print(f'{end-start}')

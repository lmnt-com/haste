<div align="center">
  <img src="https://lmnt.com/assets/haste-logo_social_media.png">
</div>

--------------------------------------------------------------------------------

Haste is a CUDA implementation of fused [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) and [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) layers with built-in [DropConnect](http://proceedings.mlr.press/v28/wan13.html) and [Zoneout](https://arxiv.org/abs/1606.01305) regularization. These layers are exposed through C++ and Python APIs for easy integration into your own projects or machine learning frameworks.

What's included in this project?
- a standalone C++ API (`libhaste`)
- a TensorFlow Python API (`haste_tf`)
- examples for writing your own custom C++ inference / training code using `libhaste`
- benchmarking programs to evaluate the performance of RNN implementations

For questions or feedback about Haste, please open an issue on GitHub or send us an email at [haste@lmnt.com](mailto:haste@lmnt.com).

## Performance
Our LSTM benchmark indicates that Haste has the fastest publicly available implementation for nearly all problem sizes.
<table>
  <tr><td><img src="https://lmnt.com/assets/haste/benchmark/report_n=16_c=128.png"></td><td><img src="https://lmnt.com/assets/haste/benchmark/report_n=32_c=256.png"></td></tr>
  <tr></tr>
  <tr><td><img src="https://lmnt.com/assets/haste/benchmark/report_n=64_c=128.png"></td><td><img src="https://lmnt.com/assets/haste/benchmark/report_n=128_c=256.png"></td></tr>
</table>

Here is our complete benchmark result grid:
<br>
[`N=1 C=64`](https://lmnt.com/assets/haste/benchmark/report_n=1_c=64.png)
[`N=1 C=128`](https://lmnt.com/assets/haste/benchmark/report_n=1_c=128.png)
[`N=1 C=256`](https://lmnt.com/assets/haste/benchmark/report_n=1_c=256.png)
[`N=1 C=512`](https://lmnt.com/assets/haste/benchmark/report_n=1_c=512.png)
<br>
[`N=32 C=64`](https://lmnt.com/assets/haste/benchmark/report_n=32_c=64.png)
[`N=32 C=128`](https://lmnt.com/assets/haste/benchmark/report_n=32_c=128.png)
[`N=32 C=256`](https://lmnt.com/assets/haste/benchmark/report_n=32_c=256.png)
[`N=32 C=512`](https://lmnt.com/assets/haste/benchmark/report_n=32_c=512.png)
<br>
[`N=64 C=64`](https://lmnt.com/assets/haste/benchmark/report_n=64_c=64.png)
[`N=64 C=128`](https://lmnt.com/assets/haste/benchmark/report_n=64_c=128.png)
[`N=64 C=256`](https://lmnt.com/assets/haste/benchmark/report_n=64_c=256.png)
[`N=64 C=512`](https://lmnt.com/assets/haste/benchmark/report_n=64_c=512.png)
<br>
[`N=128 C=64`](https://lmnt.com/assets/haste/benchmark/report_n=128_c=64.png)
[`N=128 C=128`](https://lmnt.com/assets/haste/benchmark/report_n=128_c=128.png)
[`N=128 C=256`](https://lmnt.com/assets/haste/benchmark/report_n=128_c=256.png)
[`N=128 C=512`](https://lmnt.com/assets/haste/benchmark/report_n=128_c=512.png)

## Install
Here's what you'll need to get started:
- a [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) 6.0+ GPU (required)
- [TensorFlow GPU](https://www.tensorflow.org/install/gpu) 1.14+ or 2.0+ for TensorFlow integration (optional)
- [Eigen 3](http://eigen.tuxfamily.org/) to build the C++ examples (optional)
- [cuDNN Developer Library](https://developer.nvidia.com/rdp/cudnn-archive) to build benchmarking programs (optional)

Once you have the prerequisites, run the following to build the code and install the TensorFlow API:
```
make && pip install haste_tf-*.whl
```

## Documentation
Getting started with the TensorFlow API is easy:
```python
import haste_tf as haste

lstm_layer = haste.LSTM(num_units=256, direction='bidirectional', zoneout=0.1, dropout=0.05)
gru_layer = haste.GRU(num_units=256, direction='bidirectional', zoneout=0.1, dropout=0.05)

# `x` is a tensor with shape [N,T,C]
y, state = lstm_layer(x)
y, state = gru_layer(x)
```

The TensorFlow Python API is documented in [`docs/tf/haste_tf.md`](docs/tf/haste_tf.md).
The C++ API is documented in [`lib/haste.h`](lib/haste.h) and there are code samples in [`examples/`](examples/).

## Code layout
- [`benchmarks/`](benchmarks): programs to evaluate performance of RNN implementations
- [`docs/tf/`](docs/tf): API reference documentation for `haste_tf`
- [`examples/`](examples): examples for writing your own C++ inference / training code using `libhaste`
- [`frameworks/tf/`](frameworks/tf): TensorFlow Python API and custom op code
- [`lib/`](lib): CUDA kernels and C++ API

## Implementation notes
- the GRU implementation is based on `1406.1078v1` (same as cuDNN) rather than `1406.1078v3`
- Zoneout on LSTM cells is applied to the hidden state only, and not the cell state

## References
1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. _Neural Computation_, _9_(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
1. Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. _arXiv:1406.1078 [cs, stat]_. http://arxiv.org/abs/1406.1078.
1. Wan, L., Zeiler, M., Zhang, S., Cun, Y. L., & Fergus, R. (2013). Regularization of Neural Networks using DropConnect. In _International Conference on Machine Learning_ (pp. 1058–1066). Presented at the International Conference on Machine Learning. http://proceedings.mlr.press/v28/wan13.html.
1. Krueger, D., Maharaj, T., Kramár, J., Pezeshki, M., Ballas, N., Ke, N. R., et al. (2017). Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations. _arXiv:1606.01305 [cs]_. http://arxiv.org/abs/1606.01305.
1. Ba, J., Kiros, J.R., & Hinton, G.E. (2016). Layer Normalization. _arXiv:1607.06450 [stat]_. https://arxiv.org/abs/1607.06450.

## Citing this work
To cite this work, please use the following BibTeX entry:
```
@misc{haste2020,
  title  = {Haste: a fast, simple, and open RNN library},
  author = {Sharvil Nanavati},
  year   = 2020,
  month  = "Jan",
  howpublished = {\url{https://github.com/lmnt-com/haste/}},
}
```

## License
[Apache 2.0](LICENSE)

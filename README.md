--------------------------------------------------------------------------------
[![GitHub](https://img.shields.io/github/license/lmnt-com/haste)](LICENSE)

`fast_ligru` is an open-source CUDA implementation of the [Light Gated Recurrent Units](https://arxiv.org/abs/1803.10225) that works with `PyTorch`. 
The project is modified from [Haste](https://github.com/lmnt-com/haste). 

We provide two differents implementation: `Li-GRU 1.0` and `Li-GRU 2.0`. The difference rely on the recurrent connection, in the `Li-GRU 2.0` we apply a layer normalisation to reduce the gradient exploding problem. Indeed, the `Li-GRU 1.0` is unstable and in practice cannot be trained on medium to large scale dataset (e.g, LibriSpeech 960h, CommonVoice) while the `Li-GRU 2.0` can.

For questions or feedback about `fast_ligru`, please open an issue on GitHub or send me an email at [adel.moumen@alumni.univ-avignon.fr](mailto:adel.moumen@alumni.univ-avignon.fr).

## Install
Here's what you'll need to get started:
- a [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) 3.7+ GPU (required)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0+ (required)
- [PyTorch](https://pytorch.org) 1.3+ for PyTorch integration

Once you have the prerequisites, you can install with pip or by building the source code.

### Building from source
```
make fast_ligru
```

install it with `pip`:
```
pip install fast_ligru-*.whl
```

If the CUDA Toolkit that you're building against is not in `/usr/local/cuda`, you must specify the
`$CUDA_HOME` environment variable before running make:
```
CUDA_HOME=/usr/local/cuda-10.2 make
```

## References
1. Ravanelli, M., Brakel, P., Omologo, M., & Bengio, Y. (2018). Light Gated Recurrent Units for Speech Recognition. arXiv. (https://doi.org/10.1109/TETCI.2017.2762739)

## Citing this work
To cite this work, please use the following BibTeX entry:
```
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

## License
[Apache 2.0](LICENSE)

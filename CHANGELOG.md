# ChangeLog

## 0.4.0 (2020-04-13)
### Added
- New layer normalized GRU layer (`LayerNormGRU`).
- New IndRNN layer.
- CPU support for all PyTorch layers.
- Support for building PyTorch API on Windows.
- Added `state` argument to PyTorch layers to specify initial state.
- Added weight transforms to TensorFlow API (see docs for details).
- Added `get_weights` method to extract weights from RNN layers (TensorFlow).
- Added `to_native_weights` and `from_native_weights` to PyTorch API for `LSTM` and `GRU` layers.
- Validation tests to check for correctness.

### Changed
- Performance improvements to GRU layer.
- BREAKING CHANGE: PyTorch layers default to CPU instead of GPU.
- BREAKING CHANGE: `h` must not be transposed before passing it to `gru::BackwardPass::Iterate`.

### Fixed
- Multi-GPU training with TensorFlow caused by invalid sharing of `cublasHandle_t`.

## 0.3.0 (2020-03-09)
### Added
- PyTorch support.
- New layer normalized LSTM layer (`LayerNormLSTM`).
- New fused layer normalization layer.

### Fixed
- Occasional uninitialized memory use in TensorFlow LSTM implementation.

## 0.2.0 (2020-02-12)
### Added
- New time-fused API for LSTM (`lstm::ForwardPass::Run`, `lstm::BackwardPass::Run`).
- Benchmarking code to evaluate the performance of an implementation.

### Changed
- Performance improvements to existing iterative LSTM API.
- BREAKING CHANGE: `h` must not be transposed before passing it to `lstm::BackwardPass::Iterate`.
- BREAKING CHANGE: `dv` does not need to be allocated and `v` must be passed instead to `lstm::BackwardPass::Iterate`.

## 0.1.0 (2020-01-29)
### Added
- Initial release of Haste.

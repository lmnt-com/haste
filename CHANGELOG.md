# ChangeLog

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

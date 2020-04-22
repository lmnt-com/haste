// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace haste {
namespace v0 {
namespace layer_norm_lstm {

template<typename T>
class ForwardPass {
  public:
    // training: `true` if the caller intends to perform a backward pass to compute gradients.
    // batch_size: the number of training/inference inputs provided in each tensor.
    // input_size: the dimension of each input vector.
    // hidden_size: the expected dimension of each output vector.
    // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    // Releases internal resources.
    // Blocks until all iterations have completed executing on the GPU.
    ~ForwardPass();

    // Runs the LSTM over all time steps. This method is faster than using a per-step
    // `Iterate` but requires that the entire input sequence be available upfront. In some
    // situations, this constraint may not be satisfiable (e.g. autoregressive models).
    // Users should prefer calling `Run` over `Iterate` whenever possible.
    //
    // steps: the number of iterations to run (i.e. T).
    // W: [C,H*4] the input weight matrix.
    // R: [H,H*4] the recurrent weight matrix.
    // b: [H*4] the bias vector.
    // x: [T,N,C] the LSTM input for this iteration (N vectors, each with dimension C).
    // h: [T+1,N,H] the hidden state vectors across all time steps. The t=0'th vector should
    //      be set to the desired initial hidden state (typically zeros). The rest of the
    //      vectors will be set by this function. `h[1:,:,:]` forms the output of this LSTM
    //      layer.
    // c: [T+1,N,H] the cell state vectors across all time steps. The t=0'th vector should be
    //      set to the desired initial cell state (typically zeros). The rest of the vectors
    //      will be set by this function.
    // v: [T,N,H*4] if `training` is `false`, this is scratch space and should not be used by
    //     the caller. If `training` is `true`, this parameter will contain intermediate
    //     activations which must be provided as-is to `BackwardPass::Run` or manually urolled
    //     for `BackwardPass::Iterate`.
    // tmp_Rh: [N,H*4] additional temporary work space required for this iteration. The caller
    //     should not use the contents of this vector. The same memory region may be provided
    //     for each iteration.
    // zoneout_prob: 0.0 <= zoneout_prob <= 1.0; specifies the probability of a hidden
    //     activation being randomly zoned out. If zoneout was used during training, this
    //     parameter must also be specified during inference with the same value.
    // zoneout_mask: [T,N,H] may be null to disable zoneout. This is a random binary mask
    //     following a Bernoulli(1-zoneout_prob) distribution. A different mask is typically
    //     used for each iteration.
    void Run(
        const int steps,
        const T* W,
        const T* R,
        const T* b,
        const T* x,
        T* h,
        T* c,
        T* act_Wx,
        T* tmp_Rh,
        layer_norm::ForwardPass<T>& layer_norm1,
        T* act_Wx_norm,
        T* act_Rh,
        layer_norm::ForwardPass<T>& layer_norm2,
        layer_norm::ForwardPass<T>& layer_norm3,
        T* act_c_norm,
        const float zoneout_prob,
        const T* zoneout_mask);

  private:
    void IterateInternal(
        const T* R,
        const T* b,
        const T* h,
        const T* c,
        T* h_out,
        T* c_out,
        T* v,
        T* tmp_Rh,
        T* act_Rh,
        layer_norm::ForwardPass<T>& layer_norm2,
        layer_norm::ForwardPass<T>& layer_norm3,
        T* act_c_norm,
        const float zoneout_prob,
        const T* zoneout_mask);

    struct private_data;
    private_data* data_;
};

template<typename T>
class BackwardPass {
  public:
    // batch_size: the number of training inputs provided in each tensor.
    // input_size: the dimension of each input vector.
    // hidden_size: the expected dimension of each output vector.
    // blas_handle: an initialized cuBLAS handle (see `cublasCreate`).
    BackwardPass(
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    // Releases internal resources.
    // Blocks until all iterations have completed executing on the GPU.
    ~BackwardPass();

    // Runs the LSTM backward pass over all time steps. This method is faster than using a
    // per-step `Iterate` but requires that the entire input sequence be available upfront.
    // In some situations, this constraint may not be satisfiable (e.g. autoregressive models).
    // Users should prefer calling `Run` over `Iterate` whenever possible.
    //
    // steps: the number of iterations to run (i.e. T).
    // W_t: [H*4,C] the transpose of the input weight matrix.
    // R_t: [H*4,H] the transpose of the recurrent weight matrix.
    // b: [H*4] the bias vector.
    // x_t: [C,T,N] the transpose of the LSTM input for this iteration.
    // h: [T+1,N,H] the hidden state vectors after running `ForwardPass::Run`.
    // c: [T+1,N,H] the cell state vectors after running `ForwardPass::Run`.
    // dh_new: [T+1,N,H] the gradient of the loss with respect to `h`.
    // dc_new: [T+1,N,H] the gradient of the loss with respect to `c`.
    // dx: [T,N,C] the gradient of the loss with respect to the input.
    // dW: [C,H*4] the gradient of the loss with respect to the input weight matrix.
    // dR: [H,H*4] the gradient of the loss with respect to the recurrent weight matrix.
    // db: [H*4] the gradient of the loss with respect to the bias vector.
    // dh: [N,H] NOTE: this is an input and output parameter. Should be initialized to zeros.
    //     When this function returns, `dh` will contain the gradient of the loss with respect
    //     to the initial hidden state.
    // dc: [N,H] NOTE: this is an input and output parameter. Should be initialized to zeros.
    //     When this function returns, `dc` will contain the gradient of the loss with respect
    //     to the initial cell state.
    // v: [T,N,H*4] the same tensor that was passed to `ForwardPass::Run`.
    // zoneout_mask: [T,N,H] may be null if zoneout was disabled in the forward pass. This
    //     vector must be the same as the one provided during the forward pass.
    void Run(
        const int steps,
        const T* W_t,
        const T* R_t,
        const T* b,
        const T* x_t,
        const T* h,
        const T* c,
        const T* dh_new,
        const T* dc_new,
        T* dx,
        T* dW,
        T* dR,
        T* db,
        T* dh,
        T* dc,
        T* act_Wx,
        layer_norm::BackwardPass<T>& layer_norm1,
        T* act_Wx_norm,
        T* act_Rh,
        layer_norm::BackwardPass<T>& layer_norm2,
        layer_norm::BackwardPass<T>& layer_norm3,
        T* act_c_norm,
        const T* zoneout_mask);

  private:
    void IterateInternal(
        const T* R_t,
        const T* c,
        const T* c_new,
        const T* dh_new,
        const T* dc_new,
        T* db,
        T* dh,
        T* dc,
        T* v,
        T* act_Rh,
        layer_norm::BackwardPass<T>& layer_norm2,
        layer_norm::BackwardPass<T>& layer_norm3,
        T* act_c_norm,
        const T* zoneout_mask);
    struct private_data;
    private_data* data_;
};

}  // namespace layer_norm_lstm
}  // namespace v0
}  // namespace haste

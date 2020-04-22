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
namespace gru {

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

    // Performs one forward iteration of the GRU cell.
    //
    // W: [C,H*3] the input weight matrix.
    // R: [H,H*3] the recurrent weight matrix.
    // bx: [H*3] the bias for the input weight matrix.
    // br: [H*3] the bias for the recurrent weight matrix.
    // x: [N,C] the GRU input for this iteration (N vectors, each with dimension C).
    // h: [N,H] the t-1 iteration's `h_out` or the initial hidden state if this is the
    //     t=0 iteration (typically zeros).
    // h_out: [N,H] the GRU's output, and the input to the next iteration's `h`. This
    //     pointer may be the same as `h`. Each iteration may reuse the same memory region.
    // v: [N,H*4] if `training` is `false`, this can be a null pointer. If `training` is
    //     `true`, this vector will contain intermediate activations for this iteration which
    //     must be provided as-is to the corresponding backward iteration. The caller must
    //     provide a new memory region for each iteration.
    // tmp_Wx: [N,H*3] additional temporary work space required for this iteration. The caller
    //     should not use the contents of this vector, and must provide a new memory region for
    //     each iteration.
    // tmp_Rh: [N,H*3] additional temporary work space required for this iteration. The caller
    //     should not use the contents of this vector. The same memory region may be provided
    //     for each iteration.
    // zoneout_prob: 0.0 <= zoneout_prob <= 1.0; specifies the probability of a hidden
    //     activation being randomly zoned out. If zoneout was used during training, this
    //     parameter must also be specified during inference with the same value.
    // zoneout_mask: [N,H] may be null to disable zoneout. This is a random binary mask
    //     following a Bernoulli(1-zoneout_prob) distribution. A different mask is typically
    //     used for each iteration.
    void Iterate(
        const T* W,
        const T* R,
        const T* bx,
        const T* br,
        const T* x,
        const T* h,
        T* h_out,
        T* v,
        T* tmp_Wx,
        T* tmp_Rh,
        const float zoneout_prob,
        const T* zoneout_mask);

    void Run(
        const int steps,
        const T* W,
        const T* R,
        const T* bx,
        const T* br,
        const T* x,
        T* h,
        T* v,
        T* tmp_Wx,
        T* tmp_Rh,
        const float zoneout_prob,
        const T* zoneout_mask);

  private:
    void IterateInternal(
        const T* R,
        const T* bx,
        const T* br,
        const T* h,
        T* h_out,
        T* v,
        T* tmp_Wx,
        T* tmp_Rh,
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

    // Performs one backward iteration of the GRU cell.
    //
    // Note that BackwardPass must be iterated in the reverse order as ForwardPass.
    // If ForwardPass iterates from 0 to T-1, BackwardPass needs to iterate from
    // T-1 down to 0. When iteration numbers are described, they will be based on the
    // iteration index (i.e., the T-1'th iteration of the forward pass is the last call
    // to ForwardPass::Iterate, whereas it is the first call to BackwardPass::Iterate).
    //
    // W_t: [H*3,C] the transpose of the input weight matrix.
    // R_t: [H*3,H] the transpose of the recurrent weight matrix.
    // bx: [H*3] the bias vector for the input weight matrix.
    // br: [H*3] the bias vector for the recurrent weight matrix.
    // x_t: [C,N] the transpose of the GRU input for this iteration.
    // h: [N,H] the t-1 iteration's `h_out` or the initial hidden state if this is the t=0
    //     iteration (typically zeros).
    // v: [N,H*4] the same vector as returned by ForwardPass::Iterate on its corresponding
    //     iteration.
    // dh_new: [N,H] the gradient of `h_out` with respect to the loss at this iteration.
    // dx: [N,C] the gradient of the input at this time step with respect to the loss.
    // dW: [C,H*3] the gradient of the input weight matrix with respect to the loss.
    // dR: [H,H*3] the gradient of the recurrent weight matrix with respect to the loss.
    // dbx: [H*3] the gradient of the bias vector for the input weight matrix with respect to
    //     the loss.
    // dbr: [H*3] the gradient of the bias vector for the recurrent weight matrix with respect
    //     to the loss.
    // dh: [N,H] NOTE: this is an input and output parameter. Should be initialized to zeros
    //     for the T-1'th iteration and the same pointer should be passed in for each
    //     iteration. After a complete backward pass, this vector will contain the gradient
    //     of the initial hidden state with respect to the loss.
    // dp: [N,H*3] additional temporary work space required for this iteration. The caller
    //     should not use the contents of this vector. A new memory region must be provided
    //     for each iteration.
    // dq: [N,H*3] additional temporary work space required for this iteration. The caller
    //     should not use the contents of this vector. A new memory region must be provided
    //     for each iteration.
    // zoneout_mask: [N,H] may be null if zoneout was disabled in the forward pass. This vector
    //     must be the same as the one provided during the corresponding forward iteration.
    void Iterate(
        const T* W_t,
        const T* R_t,
        const T* bx,
        const T* br,
        const T* x_t,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dx,
        T* dW,
        T* dR,
        T* dbx,
        T* dbr,
        T* dh,
        T* dp,
        T* dq,
        const T* zoneout_mask);

    void Run(
        const int steps,
        const T* W_t,
        const T* R_t,
        const T* bx,
        const T* br,
        const T* x_t,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dx,
        T* dW,
        T* dR,
        T* dbx,
        T* dbr,
        T* dh,
        T* dp,
        T* dq,
        const T* zoneout_mask);

  private:
    void IterateInternal(
        const T* R_t,
        const T* h,
        const T* v,
        const T* dh_new,
        T* dbx,
        T* dbr,
        T* dh,
        T* dp,
        T* dq,
        const T* zoneout_mask);

    struct private_data;
    private_data* data_;
};

}  // namespace gru
}  // namespace v0
}  // namespace haste

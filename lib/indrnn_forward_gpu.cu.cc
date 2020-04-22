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

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

template<typename T, bool Training, bool ApplyZoneout>
__global__
void IndrnnFwdOps(
    const int steps,
    const int batch_size,
    const int hidden_size,
    const T* Wx,
    const T* u,
    const T* b,
    const T* h,
    T* h_out,
    const float zoneout_prob,
    const T* zoneout_mask) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_size || col >= batch_size)
    return;

  const int idx = col * hidden_size + row;
  const int NH = batch_size * hidden_size;
  const T u_row = u[row];
  const T b_row = b[row];

  for (int i = 0; i < steps * NH; i += NH) {
    const T a = Wx[idx + i] + u_row * h[idx + i] + b_row;
    T cur_h_value = tanh(a);

    if (ApplyZoneout) {
      if (Training) {
        cur_h_value = (cur_h_value - h[idx + i]) * zoneout_mask[idx + i] + h[idx + i];
      } else {
        cur_h_value = (zoneout_prob * h[idx + i]) + ((1.0f - zoneout_prob) * cur_h_value);
      }
    }

    h_out[idx + i] = cur_h_value;
  }
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace indrnn {

template<typename T>
struct ForwardPass<T>::private_data {
  bool training;
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream;
  cudaStream_t sync_stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
  data_->training = training;
  data_->batch_size = batch_size;
  data_->input_size = input_size;
  data_->hidden_size = hidden_size;
  data_->blas_handle = blas_handle;
  data_->sync_stream = stream;
  cudaStreamCreate(&data_->stream);
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
  if (data_->sync_stream) {
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, data_->stream);
    cudaStreamWaitEvent(data_->sync_stream, event, 0);
    cudaEventDestroy(event);
  } else {
    cudaStreamSynchronize(data_->stream);
  }
  cudaStreamDestroy(data_->stream);
  delete data_;
}

template<typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T* W,
    const T* u,
    const T* b,
    const T* x,
    T* h,
    T* workspace,
    const float zoneout_prob,
    const T* zoneout_mask) {
  static const T alpha = static_cast<T>(1.0);
  static const T beta = static_cast<T>(0.0);

  const bool training = data_->training;
  const int batch_size = data_->batch_size;
  const int input_size = data_->input_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream = data_->stream;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  cublasSetStream(blas_handle, stream);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size, steps * batch_size, input_size,
      &alpha,
      W, hidden_size,
      x, input_size,
      &beta,
      workspace, hidden_size);

  const dim3 blockDim(64, 16);
  const dim3 gridDim(
      (hidden_size + blockDim.x - 1) / blockDim.x,
      (batch_size + blockDim.y - 1) / blockDim.y);
  const int NH = batch_size * hidden_size;
  if (training) {
    if (zoneout_prob && zoneout_mask) {
      IndrnnFwdOps<T, true, true><<<gridDim, blockDim, 0, stream>>>(
          steps,
          batch_size,
          hidden_size,
          workspace,
          u,
          b,
          h,
          h + NH,
          zoneout_prob,
          zoneout_mask);
    } else {
      IndrnnFwdOps<T, true, false><<<gridDim, blockDim, 0, stream>>>(
          steps,
          batch_size,
          hidden_size,
          workspace,
          u,
          b,
          h,
          h + NH,
          0.0f,
          nullptr);
    }
  } else {
    if (zoneout_prob && zoneout_mask) {
      IndrnnFwdOps<T, false, true><<<gridDim, blockDim, 0, stream>>>(
          steps,
          batch_size,
          hidden_size,
          workspace,
          u,
          b,
          h,
          h + NH,
          zoneout_prob,
          zoneout_mask);
    } else {
      IndrnnFwdOps<T, false, false><<<gridDim, blockDim, 0, stream>>>(
          steps,
          batch_size,
          hidden_size,
          workspace,
          u,
          b,
          h,
          h + NH,
          0.0f,
          nullptr);
    }
  }

  cublasSetStream(blas_handle, save_stream);
}

template class ForwardPass<float>;
template class ForwardPass<double>;

}  // namespace indrnn
}  // namespace v0
}  // namespace haste

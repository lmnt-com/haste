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

template<typename T, bool ApplyZoneout>
__global__
void IndrnnBwdOps(
    const int steps,
    const int batch_size,
    const int hidden_size,
    const T* u,
    const T* h_prev,
    const T* h,
    const T* dh_new,
    T* du_out,
    T* db_out,
    T* dh_inout,
    T* dk_out,
    const T* zoneout_mask) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_size || col >= batch_size)
    return;

  const int NH = batch_size * hidden_size;
  const int idx = col * hidden_size + row;

  const T u_row = u[row];
  T dh_inout_idx = dh_inout[idx];
  T du_sum = static_cast<T>(0.0);
  T db_sum = static_cast<T>(0.0);

  for (int i = (steps - 1) * NH; i >= 0; i -= NH) {
    T dh_total = dh_new[idx + i] + dh_inout_idx;
    T dh = static_cast<T>(0.0);
    if (ApplyZoneout) {
      const T mask = zoneout_mask[idx + i];
      dh = (static_cast<T>(1.0) - mask) * dh_total;
      dh_total = mask * dh_total;
    }

    const T dk = d_tanh(h[idx + i]) * dh_total;

    dk_out[idx + i] = dk;
    dh_inout_idx = dh + u_row * dk;
    du_sum += h_prev[idx + i] * dk;
    db_sum += dk;
  }

  dh_inout[idx] = dh_inout_idx;
  atomicAdd(&du_out[row], du_sum);
  atomicAdd(&db_out[row], db_sum);
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace indrnn {

template<typename T>
struct BackwardPass<T>::private_data {
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream;
  cudaStream_t sync_stream;
};

template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
  data_->batch_size = batch_size;
  data_->input_size = input_size;
  data_->hidden_size = hidden_size;
  data_->blas_handle = blas_handle;
  data_->sync_stream = stream;
  cudaStreamCreate(&data_->stream);
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
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
void BackwardPass<T>::Run(
    const int steps,
    const T* W_t,
    const T* u,
    const T* b,
    const T* x_t,
    const T* h,
    const T* dh_new,
    T* dx,
    T* dW,
    T* du,
    T* db,
    T* dh,
    T* workspace,
    const T* zoneout_mask) {
  const T alpha = static_cast<T>(1.0);
  const T beta = static_cast<T>(0.0);

  const int batch_size = data_->batch_size;
  const int input_size = data_->input_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream = data_->stream;

  const dim3 blockDim(64, 16);
  const dim3 gridDim(
      (hidden_size + blockDim.x - 1) / blockDim.x,
      (batch_size + blockDim.y - 1) / blockDim.y);
  const int NH = batch_size * hidden_size;
  if (zoneout_mask) {
    IndrnnBwdOps<T, true><<<gridDim, blockDim, 0, stream>>>(
        steps,
        batch_size,
        hidden_size,
        u,
        h,
        h + NH,
        dh_new + NH,
        du,
        db,
        dh,
        workspace,
        zoneout_mask);
  } else {
    IndrnnBwdOps<T, false><<<gridDim, blockDim, 0, stream>>>(
        steps,
        batch_size,
        hidden_size,
        u,
        h,
        h + NH,
        dh_new + NH,
        du,
        db,
        dh,
        workspace,
        nullptr);
  }

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  cublasSetStream(blas_handle, stream);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size, input_size, batch_size * steps,
      &alpha,
      workspace, hidden_size,
      x_t, batch_size * steps,
      &beta,
      dW, hidden_size);

  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      input_size, steps * batch_size, hidden_size,
      &alpha,
      W_t, input_size,
      workspace, hidden_size,
      &beta,
      dx, input_size);

  cublasSetStream(blas_handle, save_stream);
}

template class BackwardPass<float>;
template class BackwardPass<double>;

}  // namespace indrnn
}  // namespace v0
}  // namespace haste

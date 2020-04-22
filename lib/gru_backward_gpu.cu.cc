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
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const T* h,
                         const T* v,
                         const T* dh_new,
                         T* dbx_out,
                         T* dbr_out,
                         T* dh_inout,
                         T* dp_out,
                         T* dq_out,
                         const T* zoneout_mask) {  // Zoneout mask (only used if ApplyZoneout==true)
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_dim || col >= batch_dim)
    return;

  const int base_idx = col * hidden_dim + row;

  T dh_total = dh_new[base_idx] + dh_inout[base_idx];

  const int stride4_base_idx = col * (hidden_dim * 4) + row;
  const int z_idx = stride4_base_idx + 0 * hidden_dim;
  const int r_idx = stride4_base_idx + 1 * hidden_dim;
  const int g_idx = stride4_base_idx + 2 * hidden_dim;
  const int q_g_idx = stride4_base_idx + 3 * hidden_dim;

  const T z = v[z_idx];
  const T r = v[r_idx];
  const T g = v[g_idx];
  const T q_g = v[q_g_idx];

  if (ApplyZoneout) {
    const T mask = zoneout_mask[base_idx];
    dh_inout[base_idx] = (static_cast<T>(1.0) - mask) * dh_total;
    dh_total = mask * dh_total;
    dh_inout[base_idx] += z * dh_total;
  } else {
    dh_inout[base_idx] = z * dh_total;
  }

  const T dg = (static_cast<T>(1.0) - z) * dh_total;
  const T dz = (h[base_idx] - g) * dh_total;
  const T dp_g = d_tanh(g) * dg;
  const T dq_g = dp_g * r;
  const T dr = dp_g * q_g;
  const T dp_r = d_sigmoid(r) * dr;
  const T dq_r = dp_r;
  const T dp_z = d_sigmoid(z) * dz;
  const T dq_z = dp_z;

  const int idx = col * (hidden_dim * 3) + row;

  dp_out[idx + 0 * hidden_dim] = dp_z;
  dp_out[idx + 1 * hidden_dim] = dp_r;
  dp_out[idx + 2 * hidden_dim] = dp_g;

  dq_out[idx + 0 * hidden_dim] = dq_z;
  dq_out[idx + 1 * hidden_dim] = dq_r;
  dq_out[idx + 2 * hidden_dim] = dq_g;

  atomicAdd(&dbx_out[row + 0 * hidden_dim], dp_z);
  atomicAdd(&dbx_out[row + 1 * hidden_dim], dp_r);
  atomicAdd(&dbx_out[row + 2 * hidden_dim], dp_g);

  atomicAdd(&dbr_out[row + 0 * hidden_dim], dq_z);
  atomicAdd(&dbr_out[row + 1 * hidden_dim], dq_r);
  atomicAdd(&dbr_out[row + 2 * hidden_dim], dq_g);
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace gru {

template<typename T>
struct BackwardPass<T>::private_data {
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
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
  cudaStreamCreate(&data_->stream[0]);
  cudaStreamCreate(&data_->stream[1]);
  cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
  if (data_->sync_stream) {
    cudaEventRecord(data_->event, data_->stream[1]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    cudaEventRecord(data_->event, data_->stream[0]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
  } else {
    cudaStreamSynchronize(data_->stream[1]);
    cudaStreamSynchronize(data_->stream[0]);
  }
  cudaEventDestroy(data_->event);
  cudaStreamDestroy(data_->stream[1]);
  cudaStreamDestroy(data_->stream[0]);
  delete data_;
}

template<typename T>
void BackwardPass<T>::Iterate(
    const T* W_t,     // [H*3,C]
    const T* R_t,     // [H*3,H]
    const T* bx,      // [H*3]
    const T* br,      // [H*3]
    const T* x_t,     // [C,N]
    const T* h,       // [N,H]
    const T* v,       // [N,H*4]
    const T* dh_new,  // [N,H]
    T* dx,            // [N,C]
    T* dW,            // [C,H*3]
    T* dR,            // [H,H*3]
    T* dbx,           // [H*3]
    T* dbr,           // [H*3]
    T* dh,            // [N,H]
    T* dp,            // [N,H*3]
    T* dq,            // [N,H*3]
    const T* zoneout_mask) {  // [N,H]
  const T alpha = static_cast<T>(1.0);
  const T beta_sum = static_cast<T>(1.0);
  const T beta_assign = static_cast<T>(0.0);

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const int input_size = data_->input_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaStream_t stream2 = data_->stream[1];
  const cudaEvent_t event = data_->event;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  IterateInternal(
      R_t,
      h,
      v,
      dh_new,
      dbx,
      dbr,
      dh,
      dp,
      dq,
      zoneout_mask);

  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size * 3, input_size, batch_size,
      &alpha,
      dp, hidden_size * 3,
      x_t, batch_size,
      &beta_sum,
      dW, hidden_size * 3);

  // Wait for pointwise operations to complete since there's a
  // data dependency between its output (`dp`, `dq`) and the following matmuls.
  cudaStreamWaitEvent(stream2, event, 0);

  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      input_size, batch_size, hidden_size * 3,
      &alpha,
      W_t, input_size,
      dp, hidden_size * 3,
      &beta_assign,
      dx, input_size);

  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      hidden_size * 3, hidden_size, batch_size,
      &alpha,
      dq, hidden_size * 3,
      h, hidden_size,
      &beta_sum,
      dR, hidden_size * 3);

  cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void BackwardPass<T>::IterateInternal(
    const T* R_t,     // [H*3,H]
    const T* h,       // [N,H]
    const T* v,       // [N,H*4]
    const T* dh_new,  // [N,H]
    T* dbx,           // [H*3]
    T* dbr,           // [H*3]
    T* dh,            // [N,H]
    T* dp,            // [N,H*3]
    T* dq,            // [N,H*3]
    const T* zoneout_mask) {  // [N,H]
  const T alpha = static_cast<T>(1.0);
  const T beta_sum = static_cast<T>(1.0);

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaEvent_t event = data_->event;

  // Compute launch configuration for pointwise operations kernel.
  const dim3 blockDim(32, 16);
  const dim3 gridDim(
      (hidden_size + blockDim.x - 1) / blockDim.x,
      (batch_size + blockDim.y - 1) / blockDim.y);

  if (zoneout_mask) {
    PointwiseOperations<T, true><<<gridDim, blockDim, 0, stream1>>>(
        batch_size,
        hidden_size,
        h,
        v,
        dh_new,
        dbx,
        dbr,
        dh,
        dp,
        dq,
        zoneout_mask
    );
  } else {
    PointwiseOperations<T, false><<<gridDim, blockDim, 0, stream1>>>(
        batch_size,
        hidden_size,
        h,
        v,
        dh_new,
        dbx,
        dbr,
        dh,
        dp,
        dq,
        nullptr
    );
  }
  cudaEventRecord(event, stream1);

  cublasSetStream(blas_handle,  stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size, batch_size, hidden_size * 3,
      &alpha,
      R_t, hidden_size,
      dq, hidden_size * 3,
      &beta_sum,
      dh, hidden_size);
}

template<typename T>
void BackwardPass<T>::Run(
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
    const T* zoneout_mask) {
  const T alpha = static_cast<T>(1.0);
  const T beta_sum = static_cast<T>(1.0);
  const T beta_assign = static_cast<T>(0.0);

  const int batch_size = data_->batch_size;
  const int input_size = data_->input_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaStream_t stream2 = data_->stream[1];
  const cudaEvent_t event = data_->event;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  const int NH = batch_size * hidden_size;
  for (int i = steps - 1; i >= 0; --i) {
    IterateInternal(
        R_t,
        h + i * NH,
        v + i * NH * 4,
        dh_new + (i + 1) * NH,
        dbx,
        dbr,
        dh,
        dp + i * NH * 3,
        dq + i * NH * 3,
        zoneout_mask);
  }

  // Wait for pointwise operations to complete since there's a
  // data dependency between its output (`dp`, `dq`) and the following matmuls.
  cudaStreamWaitEvent(stream2, event, 0);

  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      input_size, batch_size * steps, hidden_size * 3,
      &alpha,
      W_t, input_size,
      dp, hidden_size * 3,
      &beta_assign,
      dx, input_size);

  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      hidden_size * 3, hidden_size, batch_size * steps,
      &alpha,
      dq, hidden_size * 3,
      h, hidden_size,
      &beta_sum,
      dR, hidden_size * 3);

  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size * 3, input_size, batch_size * steps,
      &alpha,
      dp, hidden_size * 3,
      x_t, batch_size * steps,
      &beta_sum,
      dW, hidden_size * 3);

  cublasSetStream(blas_handle, save_stream);
}

template struct BackwardPass<float>;
template struct BackwardPass<double>;

}  // namespace gru
}  // namespace v0
}  // namespace haste

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

#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <vector>

#include "blas.h"
#include "haste.h"
#include "inline_ops.h"

namespace {

template<typename T, bool ApplyZoneout>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const T* c,
                         const T* v,
                         const T* c_new,
                         const T* dh_new,
                         const T* dc_new,
                         T* db_out,
                         T* dh_inout,
                         T* dc_inout,
                         T* dv_out,
                         const T* zoneout_mask) {  // Zoneout mask (only used if ApplyZoneout==true)
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_dim || col >= batch_dim)
    return;

  const int base_idx = col * hidden_dim + row;

        T dc_total = dc_new[base_idx] + dc_inout[base_idx];
        T dh_total = dh_new[base_idx] + dh_inout[base_idx];
  const T c_tanh = tanh(c_new[base_idx]);

  const int stride4_base_idx = col * (hidden_dim * 4) + row;
  const int i_idx = stride4_base_idx + 0 * hidden_dim;
  const int g_idx = stride4_base_idx + 1 * hidden_dim;
  const int f_idx = stride4_base_idx + 2 * hidden_dim;
  const int o_idx = stride4_base_idx + 3 * hidden_dim;

  const T i = v[i_idx];
  const T g = v[g_idx];
  const T f = v[f_idx];
  const T o = v[o_idx];

  if (ApplyZoneout) {
    const T mask = zoneout_mask[base_idx];
    dh_inout[base_idx] = (static_cast<T>(1.0) - mask) * dh_total;
    dh_total = mask * dh_total;
  } else {
    dh_inout[base_idx] = static_cast<T>(0.0);
  }

  const T do_ = c_tanh * dh_total;
  const T dc_tanh = o * dh_total;
          dc_total += d_tanh(c_tanh) * dc_tanh;
  const T df = c[base_idx] * dc_total;
  const T dc = f * dc_total;
  const T di = g * dc_total;
  const T dg = i * dc_total;
  const T dv_g = d_tanh(g) * dg;
  const T dv_o = d_sigmoid(o) * do_;
  const T dv_i = d_sigmoid(i) * di;
  const T dv_f = d_sigmoid(f) * df;

  // TODO: performance optimization opportunity on this reduce operation.
  atomicAdd(&db_out[row + 0 * hidden_dim], dv_i);
  atomicAdd(&db_out[row + 1 * hidden_dim], dv_g);
  atomicAdd(&db_out[row + 2 * hidden_dim], dv_f);
  atomicAdd(&db_out[row + 3 * hidden_dim], dv_o);

  dc_inout[base_idx] = dc;

  dv_out[i_idx] = dv_i;
  dv_out[g_idx] = dv_g;
  dv_out[f_idx] = dv_f;
  dv_out[o_idx] = dv_o;
}

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace lstm {

template<typename T>
struct BackwardPass<T>::private_data {
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[3];
  cudaEvent_t event;
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
  cudaStreamCreate(&data_->stream[0]);
  cudaStreamCreate(&data_->stream[1]);
  cudaStreamCreate(&data_->stream[2]);
  cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
  cudaStreamSynchronize(data_->stream[2]);
  cudaStreamSynchronize(data_->stream[1]);
  cudaStreamSynchronize(data_->stream[0]);
  cudaEventDestroy(data_->event);
  cudaStreamDestroy(data_->stream[2]);
  cudaStreamDestroy(data_->stream[1]);
  cudaStreamDestroy(data_->stream[0]);
  delete data_;
}

template<typename T>
void BackwardPass<T>::Iterate(
    const T* W_t,     // [H*4,C]
    const T* R_t,     // [H*4,H]
    const T* b,       // [H*4]
    const T* x_t,     // [C,N]
    const T* h,       // [N,H]
    const T* c,       // [N,H]
    const T* c_new,   // [N,H]
    const T* dh_new,  // [N,H]
    const T* dc_new,  // [N,H]
    T* dx,            // [N,C]
    T* dW,            // [C,H*4]
    T* dR,            // [H,H*4]
    T* db,            // [H*4]
    T* dh,            // [N,H]
    T* dc,            // [N,H]
    T* v,             // [N,H*4]
    const T* zoneout_mask) {
  const T alpha = static_cast<T>(1.0);
  const T beta_sum = static_cast<T>(1.0);  // Accumulate into output matrix!
  const T beta_assign = static_cast<T>(0.0);

  const int batch_size = data_->batch_size;
  const int input_size = data_->input_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream2 = data_->stream[1];
  const cudaStream_t stream3 = data_->stream[2];
  const cudaEvent_t event = data_->event;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  IterateInternal(
      R_t,
      c,
      c_new,
      dh_new,
      dc_new,
      db,
      dh,
      dc,
      v,
      zoneout_mask);

  // Wait for pointwise operations to complete since there's a
  // data dependency between its output (`v`) and the following matmuls.
  cudaStreamWaitEvent(stream2, event, 0);
  cudaStreamWaitEvent(stream3, event, 0);

  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      input_size, batch_size, hidden_size * 4,
      &alpha,
      W_t, input_size,
      v, hidden_size * 4,
      &beta_assign,
      dx, input_size);

  cublasSetStream(blas_handle, stream3);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      hidden_size * 4, hidden_size, batch_size,
      &alpha,
      v, hidden_size * 4,
      h, hidden_size,
      &beta_sum,
      dR, hidden_size * 4);

  cublasSetStream(blas_handle, stream3);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size * 4, input_size, batch_size,
      &alpha,
      v, hidden_size * 4,
      x_t, batch_size,
      &beta_sum,
      dW, hidden_size * 4);

  cublasSetStream(blas_handle, save_stream);
}

template<typename T>
void BackwardPass<T>::IterateInternal(
    const T* R_t,     // [H*4,H]
    const T* c,       // [N,H]
    const T* c_new,   // [N,H]
    const T* dh_new,  // [N,H]
    const T* dc_new,  // [N,H]
    T* db,            // [H*4]
    T* dh,            // [N,H]
    T* dc,            // [N,H]
    T* v,             // [N,H*4]
    const T* zoneout_mask) {
  const T alpha = static_cast<T>(1.0);
  const T beta_sum = static_cast<T>(1.0);  // Accumulate into output matrix!

  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaEvent_t event = data_->event;

  // Compute launch configuration for pointwise operations kernel.
  const dim3 blockDim(64, 16);
  const dim3 gridDim(
      (hidden_size + blockDim.x - 1) / blockDim.x,
      (batch_size + blockDim.y - 1) / blockDim.y);

  if (zoneout_mask) {
    PointwiseOperations<T, true><<<gridDim, blockDim, 0, stream1>>>(
        batch_size,
        hidden_size,
        c,
        v,
        c_new,
        dh_new,
        dc_new,
        db,
        dh,
        dc,
        v,
        zoneout_mask
    );
  } else {
    PointwiseOperations<T, false><<<gridDim, blockDim, 0, stream1>>>(
        batch_size,
        hidden_size,
        c,
        v,
        c_new,
        dh_new,
        dc_new,
        db,
        dh,
        dc,
        v,
        nullptr
    );
  }

  // Signal completion of pointwise operations for data-dependent streams.
  cudaEventRecord(event, stream1);

  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size, batch_size, hidden_size * 4,
      &alpha,
      R_t, hidden_size,
      v, hidden_size * 4,
      &beta_sum,
      dh, hidden_size);
}

template<typename T>
void BackwardPass<T>::Run(
    const int steps,
    const T* W_t,     // [H*4,C]
    const T* R_t,     // [H*4,H]
    const T* b,       // [H*4]
    const T* x_t,     // [C,T,N]
    const T* h,       // [T+1,N,H]
    const T* c,       // [T+1,N,H]
    const T* dh_new,  // [T+1,N,H]
    const T* dc_new,  // [T+1,N,H]
    T* dx,            // [T,N,C]
    T* dW,            // [C,H*4]
    T* dR,            // [H,H*4]
    T* db,            // [H*4]
    T* dh,            // [N,H]
    T* dc,            // [N,H]
    T* v,            // [T,N,H*4]
    const T* zoneout_mask) {
  const T alpha = static_cast<T>(1.0);
  const T beta_sum = static_cast<T>(1.0);  // Accumulate into output matrix!
  const T beta_assign = static_cast<T>(0.0);

  const int batch_size = data_->batch_size;
  const int input_size = data_->input_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaStream_t stream2 = data_->stream[1];
  const cudaStream_t stream3 = data_->stream[2];
  const cudaEvent_t event = data_->event;

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  const int NH = batch_size * hidden_size;
  for (int i = steps - 1; i >= 0; --i) {
    IterateInternal(
        R_t,
        c + i * NH,
        c + (i + 1) * NH,
        dh_new + (i + 1) * NH,
        dc_new + (i + 1) * NH,
        db,
        dh,
        dc,
        v + i * NH * 4,
        zoneout_mask ? zoneout_mask + i * NH : nullptr);
  }
  cudaEventRecord(event, stream1);

  cudaStreamWaitEvent(stream2, event, 0);
  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size * 4, input_size, batch_size * steps,
      &alpha,
      v, hidden_size * 4,
      x_t, batch_size * steps,
      &beta_sum,
      dW, hidden_size * 4);

  cudaStreamWaitEvent(stream3, event, 0);
  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      hidden_size * 4, hidden_size, batch_size * steps,
      &alpha,
      v, hidden_size * 4,
      h, hidden_size,
      &beta_sum,
      dR, hidden_size * 4);

  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      input_size, steps * batch_size, hidden_size * 4,
      &alpha,
      W_t, input_size,
      v, hidden_size * 4,
      &beta_assign,
      dx, input_size);

  cublasSetStream(blas_handle, save_stream);
}

template struct BackwardPass<float>;
template struct BackwardPass<double>;

}  // namespace lstm
}  // namespace v0
}  // namespace haste

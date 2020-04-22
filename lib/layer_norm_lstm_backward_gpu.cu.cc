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
void ComputeOutputGrad(
    const int batch_size,
    const int hidden_size,
    const T* act_c_norm,
    const T* dh_new,
    T* dh_inout,
    T* dlayer_norm,
    T* v,
    const T* zoneout_mask) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_size || col >= batch_size)
    return;

  const int base_idx = col * hidden_size + row;
  const int stride4_base_idx = col * (hidden_size * 4) + row;
  const int o_idx = stride4_base_idx + 3 * hidden_size;

  T dh_total = dh_new[base_idx] + dh_inout[base_idx];
  if (ApplyZoneout) {
    const T mask = zoneout_mask[base_idx];
    dh_inout[base_idx] = (static_cast<T>(1.0) - mask) * dh_total;
    dh_total = mask * dh_total;
  } else {
    dh_inout[base_idx] = static_cast<T>(0.0);
  }

  const T c_tanh = tanh(act_c_norm[base_idx]);
  const T o = v[o_idx];

  const T do_ = c_tanh * dh_total;
  const T dc_tanh = o * dh_total;

  dlayer_norm[base_idx] = d_tanh(c_tanh) * dc_tanh;
  v[o_idx] = d_sigmoid(o) * do_;
}

template<typename T>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const T* c,
                         const T* v,
                         const T* dc_new,
                         const T* dlayer_norm,
                         T* db_out,
                         T* dc_inout,
                         T* dv_out) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_dim || col >= batch_dim)
    return;

  const int base_idx = col * hidden_dim + row;
  const int stride4_base_idx = col * (hidden_dim * 4) + row;
  const int i_idx = stride4_base_idx + 0 * hidden_dim;
  const int g_idx = stride4_base_idx + 1 * hidden_dim;
  const int f_idx = stride4_base_idx + 2 * hidden_dim;
  const int o_idx = stride4_base_idx + 3 * hidden_dim;

  const T i = v[i_idx];
  const T g = v[g_idx];
  const T f = v[f_idx];
  const T o = v[o_idx];

  const T dc_total = dc_new[base_idx] + dc_inout[base_idx] + dlayer_norm[base_idx];
  const T df = c[base_idx] * dc_total;
  const T dc = f * dc_total;
  const T di = g * dc_total;
  const T dg = i * dc_total;
  const T dv_g = d_tanh(g) * dg;
  const T dv_o = o;
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
namespace layer_norm_lstm {

template<typename T>
struct BackwardPass<T>::private_data {
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[3];
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
  cudaStreamCreate(&data_->stream[2]);
  cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
BackwardPass<T>::~BackwardPass() {
  if (data_->sync_stream) {
    cudaEventRecord(data_->event, data_->stream[2]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    cudaEventRecord(data_->event, data_->stream[1]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    cudaEventRecord(data_->event, data_->stream[0]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
  } else {
    cudaStreamSynchronize(data_->stream[2]);
    cudaStreamSynchronize(data_->stream[1]);
    cudaStreamSynchronize(data_->stream[0]);
  }
  cudaEventDestroy(data_->event);
  cudaStreamDestroy(data_->stream[2]);
  cudaStreamDestroy(data_->stream[1]);
  cudaStreamDestroy(data_->stream[0]);
  delete data_;
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
    T* act_Rh,
    layer_norm::BackwardPass<T>& layer_norm2,
    layer_norm::BackwardPass<T>& layer_norm3,
    T* act_c_norm,
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
    ComputeOutputGrad<T, true><<<gridDim, blockDim, 0, stream1>>>(
        batch_size,
        hidden_size,
        act_c_norm,
        dh_new,
        dh,
        act_c_norm,
        v,
        zoneout_mask);
  } else {
    ComputeOutputGrad<T, false><<<gridDim, blockDim, 0, stream1>>>(
        batch_size,
        hidden_size,
        act_c_norm,
        dh_new,
        dh,
        act_c_norm,
        v,
        nullptr);
  }
  layer_norm3.RunPartial(stream1, batch_size, act_c_norm, act_c_norm);
  PointwiseOperations<T><<<gridDim, blockDim, 0, stream1>>>(
      batch_size,
      hidden_size,
      c,
      v,
      dc_new,
      act_c_norm,
      db,
      dc,
      v);

  // Signal completion of pointwise operations for data-dependent streams.
  cudaEventRecord(event, stream1);

  cublasSetStream(blas_handle, stream1);
  layer_norm2.RunPartial(stream1, batch_size, v, act_Rh);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size, batch_size, hidden_size * 4,
      &alpha,
      R_t, hidden_size,
      act_Rh, hidden_size * 4,
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
    T* act_Wx,        // [T,N,H*4]
    layer_norm::BackwardPass<T>& layer_norm1,
    T* act_Wx_norm,   // [T,N,H*4]
    T* act_Rh,
    layer_norm::BackwardPass<T>& layer_norm2,
    layer_norm::BackwardPass<T>& layer_norm3,
    T* act_c_norm,
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
        act_Wx_norm + i * NH * 4,
        act_Rh + i * NH * 4,
        layer_norm2,
        layer_norm3,
        act_c_norm + i * NH,
        zoneout_mask ? zoneout_mask + i * NH : nullptr);
  }
  cudaEventRecord(event, stream1);

  cudaStreamWaitEvent(stream2, event, 0);
  layer_norm1.Run(stream2, act_Wx_norm, act_Wx);
  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      hidden_size * 4, input_size, batch_size * steps,
      &alpha,
      act_Wx, hidden_size * 4,
      x_t, batch_size * steps,
      &beta_sum,
      dW, hidden_size * 4);

  cudaStreamWaitEvent(stream3, event, 0);
  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_T,
      hidden_size * 4, hidden_size, batch_size * steps,
      &alpha,
      act_Rh, hidden_size * 4,
      h, hidden_size,
      &beta_sum,
      dR, hidden_size * 4);

  cublasSetStream(blas_handle, stream2);
  blas<T>::gemm(blas_handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      input_size, steps * batch_size, hidden_size * 4,
      &alpha,
      W_t, input_size,
      act_Wx, hidden_size * 4,
      &beta_assign,
      dx, input_size);

  cublasSetStream(blas_handle, save_stream);
}

template struct BackwardPass<float>;
template struct BackwardPass<double>;

}  // namespace layer_norm_lstm
}  // namespace v0
}  // namespace haste

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

// `c` and `c_out` may be aliased.
template <typename T, bool Training>
__global__ void ComputeCellState(const int batch_size, const int hidden_size,
                                 const T *Wx, // Precomputed (Wx) vector
                                 const T *Rh, // Precomputed (Rh) vector
                                 const T *b,  // Bias for gates
                                 const T *c,  // Input cell state
                                 T *c_out,    // Output cell state
                                 T *v_out) {  // Output vector v (Wx + Rh + b)
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_size || col >= batch_size)
    return;

  // Base index into the Wx and Rh matrices.
  const int weight_idx = col * (hidden_size * 4) + row;

  // Base index into the output matrix. This is different from `weight_idx`
  // because the number of rows are different between the two sets of matrices.
  const int output_idx = col * hidden_size + row;

  const int i_idx = weight_idx + 0 * hidden_size;
  const int g_idx = weight_idx + 1 * hidden_size;
  const int f_idx = weight_idx + 2 * hidden_size;
  const int o_idx = weight_idx + 3 * hidden_size;

  const T i = sigmoid(Wx[i_idx] + Rh[i_idx] + b[row + 0 * hidden_size]);
  const T g = tanh(Wx[g_idx] + Rh[g_idx] + b[row + 1 * hidden_size]);
  const T f = sigmoid(Wx[f_idx] + Rh[f_idx] + b[row + 2 * hidden_size]);
  const T o = sigmoid(Wx[o_idx] + Rh[o_idx] + b[row + 3 * hidden_size]);

  // Compile-time constant branch should be eliminated by compiler so we have
  // straight-through code.
  if (Training) {
    v_out[i_idx] = i;
    v_out[g_idx] = g;
    v_out[f_idx] = f;
    v_out[o_idx] = o;
  } else {
    v_out[o_idx] = o;
  }

  c_out[output_idx] = (f * c[output_idx]) + (i * g);
}

// `h` and `h_out` may be aliased.
template <typename T, bool Training, bool ApplyZoneout>
__global__ void ComputeCellOutput(
    const int batch_size, const int hidden_size,
    const T *h, // Input recurrent state
    const T *c, // Input cell state
    const T *v,
    T *h_out, // Output recurrent state
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask (only used if ApplyZoneout==true)
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_size || col >= batch_size)
    return;

  const int weight_idx = col * (hidden_size * 4) + row;
  const int output_idx = col * hidden_size + row;

  const T o = v[weight_idx + 3 * hidden_size];
  const T cur_c_value = c[output_idx];

  T cur_h_value = o * tanh(cur_c_value);

  if (ApplyZoneout) {
    if (Training) {
      cur_h_value = (cur_h_value - h[output_idx]) * zoneout_mask[output_idx] +
                    h[output_idx];
    } else {
      cur_h_value = (zoneout_prob * h[output_idx]) +
                    ((1.0f - zoneout_prob) * cur_h_value);
    }
  }

  h_out[output_idx] = cur_h_value;
}

} // anonymous namespace

namespace haste {
namespace v0 {
namespace layer_norm_lstm {

template <typename T> struct ForwardPass<T>::private_data {
  bool training;
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
  cudaStream_t sync_stream;
};

template <typename T>
ForwardPass<T>::ForwardPass(const bool training, const int batch_size,
                            const int input_size, const int hidden_size,
                            const cublasHandle_t &blas_handle,
                            const cudaStream_t &stream)
    : data_(new private_data) {
  data_->training = training;
  data_->batch_size = batch_size;
  data_->input_size = input_size;
  data_->hidden_size = hidden_size;
  data_->blas_handle = blas_handle;
  data_->sync_stream = stream;
  cudaStreamCreate(&data_->stream[0]);
  cudaStreamCreate(&data_->stream[1]);
  cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template <typename T> ForwardPass<T>::~ForwardPass() {
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

template <typename T>
void ForwardPass<T>::IterateInternal(
    const T *R, // Weight matrix for recurrent state (Rh) [H,H*4]
    const T *b, // Bias for gates (Wx + Rh + b) [H*4]
    const T *h, // Recurrent state [N,H]
    const T *c, // Cell state [N,H]
    T *h_out,   // Output recurrent state [N,H]
    T *c_out,   // Output cell state [N,H]
    T *v,       // Output vector (Wx + Rh + b) [N,H*4]
    T *tmp_Rh,  // Temporary storage for Rh vector [N,H*4]
    T *act_Rh, layer_norm::ForwardPass<T> &layer_norm2,
    layer_norm::ForwardPass<T> &layer_norm3, T *act_c_norm,
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask [N,H]
  static const T alpha = static_cast<T>(1.0);
  static const T beta = static_cast<T>(0.0);

  const bool training = data_->training;
  const int batch_size = data_->batch_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];
  const cudaEvent_t event = data_->event;

  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 4,
                batch_size, hidden_size, &alpha, R, hidden_size * 4, h,
                hidden_size, &beta, act_Rh, hidden_size * 4);
  layer_norm2.RunPartial(stream1, batch_size, act_Rh, tmp_Rh);
  cudaStreamWaitEvent(stream1, event, 0);

  // Compute launch configuration for pointwise operations kernel.
  const dim3 blockDim(64, 16);
  const dim3 gridDim((hidden_size + blockDim.x - 1) / blockDim.x,
                     (batch_size + blockDim.y - 1) / blockDim.y);

  if (training) {
    ComputeCellState<T, true><<<gridDim, blockDim, 0, stream1>>>(
        batch_size, hidden_size, v, tmp_Rh, b, c, c_out, v);
    layer_norm3.RunPartial(stream1, batch_size, c_out, act_c_norm);
    if (zoneout_prob && zoneout_mask) {
      ComputeCellOutput<T, true, true><<<gridDim, blockDim, 0, stream1>>>(
          batch_size, hidden_size, h, act_c_norm, v, h_out, zoneout_prob,
          zoneout_mask);
    } else {
      ComputeCellOutput<T, true, false><<<gridDim, blockDim, 0, stream1>>>(
          batch_size, hidden_size, h, act_c_norm, v, h_out, 0.0f, nullptr);
    }
  } else {
    ComputeCellState<T, false><<<gridDim, blockDim, 0, stream1>>>(
        batch_size, hidden_size, v, tmp_Rh, b, c, c_out, v);
    layer_norm3.RunPartial(stream1, batch_size, c_out, act_c_norm);
    if (zoneout_prob && zoneout_mask) {
      ComputeCellOutput<T, false, true><<<gridDim, blockDim, 0, stream1>>>(
          batch_size, hidden_size, h, act_c_norm, v, h_out, zoneout_prob,
          zoneout_mask);
    } else {
      ComputeCellOutput<T, false, false><<<gridDim, blockDim, 0, stream1>>>(
          batch_size, hidden_size, h, act_c_norm, v, h_out, 0.0f, nullptr);
    }
  }
}

template <typename T>
void ForwardPass<T>::Run(
    const int steps,
    const T *W, // Weight matrix for input (Wx) [C,H*4]
    const T *R, // Weight matrix for recurrent state (Rh) [H,H*4]
    const T *b, // Bias for gates (Wx + Rh + b) [H*4]
    const T *x, // Input vector [T,N,C]
    T *h,       // Recurrent state [T+1,N,H]
    T *c,       // Cell state [T+1,N,H]
    T *act_Wx,  // Output vector (Wx + Rh + b) [T,N,H*4]
    T *tmp_Rh,  // Temporary storage for Rh vector [N,H*4]
    layer_norm::ForwardPass<T> &layer_norm1, T *act_Wx_norm, T *act_Rh,
    layer_norm::ForwardPass<T> &layer_norm2,
    layer_norm::ForwardPass<T> &layer_norm3, T *act_c_norm,
    const float zoneout_prob,
    const T *zoneout_mask) { // Zoneout mask [T,N,H]
  static const T alpha = static_cast<T>(1.0);
  static const T beta = static_cast<T>(0.0);

  const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

  const int batch_size = data_->batch_size;
  const int input_size = data_->input_size;
  const int hidden_size = data_->hidden_size;
  const cublasHandle_t blas_handle = data_->blas_handle;
  const cudaStream_t stream1 = data_->stream[0];

  cudaStream_t save_stream;
  cublasGetStream(blas_handle, &save_stream);

  cublasSetStream(blas_handle, stream1);
  blas<T>::gemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden_size * 4,
                steps * batch_size, input_size, &alpha, W, hidden_size * 4, x,
                input_size, &beta, act_Wx, hidden_size * 4);
  layer_norm1.Run(stream1, act_Wx, act_Wx_norm);

  for (int i = 0; i < steps; ++i) {
    const int NH = batch_size * hidden_size;
    IterateInternal(R, b, h + i * NH, c + i * NH, h + (i + 1) * NH,
                    c + (i + 1) * NH, act_Wx_norm + i * NH * 4, tmp_Rh,
                    act_Rh + i * NH * 4, layer_norm2, layer_norm3,
                    act_c_norm + i * NH, zoneout_prob,
                    zoneout_mask ? zoneout_mask + i * NH : nullptr);
  }

  cublasSetStream(blas_handle, save_stream);
}

template struct ForwardPass<float>;
template struct ForwardPass<double>;

} // namespace layer_norm_lstm
} // namespace v0
} // namespace haste

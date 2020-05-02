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

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace haste {
namespace v0 {
namespace layer_norm_indrnn {

template<typename T>
class ForwardPass {
  public:
    ForwardPass(
        const bool training,
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    ~ForwardPass();

    void Run(
        const int steps,
        const T* W,
        const T* u,
        const T* b,
        const T* x,
        T* h,
        T* workspace,
        T* act_Wx,
        layer_norm::ForwardPass<T>& layer_norm1,
        const float zoneout_prob,
        const T* zoneout_mask);

  private:
    struct private_data;
    private_data* data_;
};

template<typename T>
class BackwardPass {
  public:
    BackwardPass(
        const int batch_size,
        const int input_size,
        const int hidden_size,
        const cublasHandle_t& blas_handle,
        const cudaStream_t& stream = 0);

    ~BackwardPass();

    void Run(
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
        layer_norm::BackwardPass<T>& layer_norm1,
        const T* zoneout_mask);

  private:
    struct private_data;
    private_data* data_;
};

}  // namespace layer_norm_indrnn
}  // namespace v0
}  // namespace haste

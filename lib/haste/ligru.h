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
namespace ligru {

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

    void Iterate(
        const T* w,
        const T* u,
        const T* x,
        const T* h,
        T* h_out,
        T* v,
        T* tmp_wx,
        T* tmp_uh,
        const T* drop_mask);

    void Run(
        const int time_step,
        const T* w,
        const T* u,
        const T* x,
        T* h,
        T* v,
        T* tmp_wx,
        T* tmp_uh,
        const T* drop_mask);


  private:
    void IterateInternal(
        const T* u,
        const T* h,
        T* h_out,
        T* v,
        T* tmp_wx,
        T* tmp_uh,
        const T* drop_mask);

    struct private_data;
    private_data* data_;
};

}  // namespace ligru
}  // namespace v0
}  // namespace haste

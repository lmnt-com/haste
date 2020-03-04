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

namespace haste {
namespace v0 {
namespace layer_norm {

template<typename T>
class ForwardPass {
  public:
    // gamma: [H]
    // beta: [H]
    // cache: [N,2]
    ForwardPass(
        const int batch_size,
        const int hidden_size,
        const T* gamma,
        const T* beta,
        T* cache);

    // Computes the layer norm of an input tensor `x` over its innermost (fastest changing)
    // dimension. The layer norm is defined as: \(\frac{x-\mu}{\sigma} \gamma + \beta\)
    // where `\gamma` and `\beta` are trainable parameters.
    //
    // x: [N,H]
    // y: [N,H]
    void Run(const cudaStream_t& stream, const T* x, T* y);

    void RunPartial(
        const cudaStream_t& stream,
        const int minibatch,
        const T* x,
        T* y);

  private:
    const int batch_size_;
    const int hidden_size_;
    const T* gamma_;
    const T* beta_;
    T* cache_;
    int partial_;
};

template<typename T>
class BackwardPass {
  public:
    BackwardPass(
        const int batch_size,
        const int hidden_size,
        const T* gamma,
        const T* beta,
        const T* x,
        T* dgamma,
        T* dbeta,
        T* cache);

    void Run(const cudaStream_t& stream, const T* dy, T* dx);

    void RunPartial(
        const cudaStream_t& stream,
        const int minibatch,
        const T* dy,
        T* dx);

  private:
    const int batch_size_;
    const int hidden_size_;
    const T* gamma_;
    const T* beta_;
    const T* x_;
    T* dgamma_;
    T* dbeta_;
    T* cache_;
    int partial_;
};

}  // namespace layer_norm
}  // namespace v0
}  // namespace haste

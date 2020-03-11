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

namespace tensorflow {
class OpKernelContext;
}

#define REGISTER_GPU_KERNEL(NAME, T)                 \
  REGISTER_KERNEL_BUILDER(Name(#NAME)                \
                            .Device(DEVICE_GPU)      \
                            .TypeConstraint<T>("R"), \
                          NAME##Op<T>)

cublasHandle_t GetCublasHandle(tensorflow::OpKernelContext* context);
const cudaStream_t& GetCudaStream(tensorflow::OpKernelContext* context);

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

#include <vector>

#include "support.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/stream.h"

// LOL.
struct CublasHandleContainer {
  CublasHandleContainer() {
    int count;
    int current_device;
    cudaGetDevice(&current_device);
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; ++i) {
      cublasHandle_t handle;
      cudaSetDevice(i);
      cublasCreate(&handle);
      handles.push_back(handle);
    }
    cudaSetDevice(current_device);
  }

  ~CublasHandleContainer() {
    for (auto h : handles)
      cublasDestroy(h);
  }

  std::vector<cublasHandle_t> handles;
};

cublasHandle_t GetCublasHandle() {
  static CublasHandleContainer all_handles;
  int device;
  cudaGetDevice(&device);
  return all_handles.handles[device];
}

const cudaStream_t& GetCudaStream(tensorflow::OpKernelContext* context) {
  const auto ptr =
      context->op_device_context()->stream()->implementation()->GpuStreamMemberHack();
  return *reinterpret_cast<const cudaStream_t*>(ptr);
}

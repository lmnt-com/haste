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

#include <iterator>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "support.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/stream.h"

// LOL.
cublasHandle_t GetCublasHandle(tensorflow::OpKernelContext* context) {
  static std::unordered_map<std::thread::id, cublasHandle_t> handle_map;
  static std::mutex mutex;

  std::lock_guard<std::mutex> lock(mutex);
  std::thread::id tid = std::this_thread::get_id();
  cudaStream_t stream = GetCudaStream(context);
  auto i = handle_map.find(tid);
  if (i == std::end(handle_map)) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    i = handle_map.insert(std::make_pair(tid, handle)).first;
  }

  return i->second;
}

const cudaStream_t& GetCudaStream(tensorflow::OpKernelContext* context) {
  const auto ptr =
      context->op_device_context()->stream()->implementation()->GpuStreamMemberHack();
  return *reinterpret_cast<const cudaStream_t*>(ptr);
}

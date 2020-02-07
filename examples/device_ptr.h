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

#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>

template<typename T>
struct device_ptr {
  static constexpr size_t ElemSize = sizeof(typename T::Scalar);

  static device_ptr<T> NewByteSized(size_t bytes) {
    return device_ptr<T>((bytes + ElemSize - 1) / ElemSize);
  }

  explicit device_ptr(size_t size_)
      : data(nullptr), size(size_) {
    void* tmp;
    cudaMalloc(&tmp, size * ElemSize);
    data = static_cast<typename T::Scalar*>(tmp);
  }

  explicit device_ptr(const T& elem)
      : data(nullptr), size(elem.size()) {
    void* tmp;
    cudaMalloc(&tmp, size * ElemSize);
    data = static_cast<typename T::Scalar*>(tmp);
    ToDevice(elem);
  }

  device_ptr(device_ptr<T>&& other) : data(other.data), size(other.size) {
    other.data = nullptr;
    other.size = 0;
  }

  device_ptr& operator=(const device_ptr<T>&& other) {
    if (&other != this) {
      data = other.data;
      size = other.size;
      other.data = nullptr;
      other.size = 0;
    }
    return *this;
  }

  device_ptr(const device_ptr<T>& other) = delete;
  device_ptr& operator=(const device_ptr<T>& other) = delete;

  void ToDevice(const T& src) {
    assert(size == src.size());
    cudaMemcpy(data, src.data(), src.size() * ElemSize, cudaMemcpyHostToDevice);
  }

  void ToHost(T& target) const {
    assert(size == target.size());
    cudaMemcpy(target.data(), data, target.size() * ElemSize, cudaMemcpyDeviceToHost);
  }

  size_t Size() const {
    return size;
  }

  void zero() {
    cudaMemset(data, 0, size * ElemSize);
  }

  ~device_ptr() {
    cudaFree(data);
  }

  typename T::Scalar* data;
  size_t size;
};

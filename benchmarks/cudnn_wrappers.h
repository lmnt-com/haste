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
#include <cudnn.h>
#include <vector>

template<typename T>
struct CudnnDataType {};

template<>
struct CudnnDataType<float> {
  static constexpr auto value = CUDNN_DATA_FLOAT;
};

template<>
struct CudnnDataType<double> {
  static constexpr auto value = CUDNN_DATA_DOUBLE;
};

template<typename T>
class TensorDescriptor {
  public:
    TensorDescriptor(const std::vector<int>& dims) {
      std::vector<int> strides;
      int stride = 1;
      for (int i = dims.size() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= dims[i];
      }
      cudnnCreateTensorDescriptor(&descriptor_);
      cudnnSetTensorNdDescriptor(descriptor_, CudnnDataType<T>::value, dims.size(), &dims[0], &strides[0]);
    }

    ~TensorDescriptor() {
      cudnnDestroyTensorDescriptor(descriptor_);
    }

    cudnnTensorDescriptor_t& operator*() {
      return descriptor_;
    }

  private:
    cudnnTensorDescriptor_t descriptor_;
};


template<typename T>
class TensorDescriptorArray {
  public:
    TensorDescriptorArray(int count, const std::vector<int>& dims) {
      std::vector<int> strides;
      int stride = 1;
      for (int i = dims.size() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= dims[i];
      }
      for (int i = 0; i < count; ++i) {
        cudnnTensorDescriptor_t descriptor;
        cudnnCreateTensorDescriptor(&descriptor);
        cudnnSetTensorNdDescriptor(descriptor, CudnnDataType<T>::value, dims.size(), &dims[0], &strides[0]);
        descriptors_.push_back(descriptor);
      }
    }

    ~TensorDescriptorArray() {
      for (auto& desc : descriptors_)
        cudnnDestroyTensorDescriptor(desc);
    }

    cudnnTensorDescriptor_t* operator&() {
      return &descriptors_[0];
    }

  private:
    std::vector<cudnnTensorDescriptor_t> descriptors_;
};

class DropoutDescriptor {
  public:
    DropoutDescriptor(const cudnnHandle_t& handle) {
      cudnnCreateDropoutDescriptor(&descriptor_);
      cudnnSetDropoutDescriptor(descriptor_, handle, 0.0f, nullptr, 0, 0LL);
    }

    ~DropoutDescriptor() {
      cudnnDestroyDropoutDescriptor(descriptor_);
    }

    cudnnDropoutDescriptor_t& operator*() {
      return descriptor_;
    }

  private:
    cudnnDropoutDescriptor_t descriptor_;
};

template<typename T>
class RnnDescriptor {
  public:
    RnnDescriptor(const cudnnHandle_t& handle, int size, cudnnRNNMode_t algorithm) : dropout_(handle) {
      cudnnCreateRNNDescriptor(&descriptor_);
      cudnnSetRNNDescriptor(
          handle,
          descriptor_,
          size,
          1,
          *dropout_,
          CUDNN_LINEAR_INPUT,
          CUDNN_UNIDIRECTIONAL,
          algorithm,
          CUDNN_RNN_ALGO_STANDARD,
          CudnnDataType<T>::value);
    }

    ~RnnDescriptor() {
      cudnnDestroyRNNDescriptor(descriptor_);
    }

    cudnnRNNDescriptor_t& operator*() {
      return descriptor_;
    }

  private:
    cudnnRNNDescriptor_t descriptor_;
    DropoutDescriptor dropout_;
};

template<typename T>
class FilterDescriptor {
  public:
    FilterDescriptor(const size_t size) {
      int filter_dim[] = { (int)size, 1, 1 };
      cudnnCreateFilterDescriptor(&descriptor_);
      cudnnSetFilterNdDescriptor(descriptor_, CudnnDataType<T>::value, CUDNN_TENSOR_NCHW, 3, filter_dim);
    }

    ~FilterDescriptor() {
      cudnnDestroyFilterDescriptor(descriptor_);
    }

    cudnnFilterDescriptor_t& operator*() {
      return descriptor_;
    }

  private:
    cudnnFilterDescriptor_t descriptor_;
};

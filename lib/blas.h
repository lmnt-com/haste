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

template <typename T> struct blas {
  struct set_pointer_mode {
    set_pointer_mode(cublasHandle_t handle) : handle_(handle) {
      cublasGetPointerMode(handle_, &old_mode_);
      cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
    }
    ~set_pointer_mode() { cublasSetPointerMode(handle_, old_mode_); }

  private:
    cublasHandle_t handle_;
    cublasPointerMode_t old_mode_;
  };
  struct enable_tensor_cores {
    enable_tensor_cores(cublasHandle_t handle) : handle_(handle) {
      cublasGetMathMode(handle_, &old_mode_);
      cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
    }
    ~enable_tensor_cores() { cublasSetMathMode(handle_, old_mode_); }

  private:
    cublasHandle_t handle_;
    cublasMath_t old_mode_;
  };
};

template <> struct blas<__half> {
  static constexpr decltype(cublasHgemm) *gemm = &cublasHgemm;
};

template <> struct blas<float> {
  static constexpr decltype(cublasSgemm) *gemm = &cublasSgemm;
};

template <> struct blas<double> {
  static constexpr decltype(cublasDgemm) *gemm = &cublasDgemm;
};

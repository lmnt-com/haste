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
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

template<typename T>
class TensorView {
  public:
    TensorView(T* ptr, const tensorflow::TensorShape& shape) {
      ptr_ = ptr;
      leading_dim_ = shape.dim_size(0);
      stride_ = 1LL;
      for (int i = 1; i < shape.dims(); ++i)
        stride_ *= shape.dim_size(i);
    }

    tensorflow::int64 AllocatedBytes() const {
      return leading_dim_ * stride_ * sizeof(T);
    }

    T* data() {
      return ptr_;
    }

    T* SubSlicePtr(tensorflow::int64 i) {
      assert(i >= 0);
      assert(i < leading_dim_);

      return ptr_ + (i * stride_);
    }

    const tensorflow::int64 NumElements() const {
      return leading_dim_ * stride_;
    }

  private:
    T* ptr_;
    tensorflow::int64 leading_dim_;
    tensorflow::int64 stride_;
};

template<typename T>
class Arena {
  public:
    struct Entry {
      std::string name;
      TensorView<T> view;
    };

    Arena(std::initializer_list<Entry> map) : map_(map) {}
    Arena(const std::vector<Entry>& map) : map_(map) {}

    TensorView<T> operator[](const std::string& name) {
      for (auto& unit : map_)
        if (name == unit.name)
          return unit.view;
      assert(false && "Invalid tensor name.");
    }

  private:
    std::vector<Entry> map_;
};

template<typename T>
class ArenaLayout {
  public:
    struct Entry {
      std::string name;
      tensorflow::TensorShape shape;
    };

    ArenaLayout(std::initializer_list<Entry> layout) : layout_(layout) {}

    tensorflow::int64 NumElements() const {
      tensorflow::int64 total_elements = 0;
      for (const auto& entry : layout_)
        total_elements += entry.shape.num_elements();
      return total_elements;
    }

    Arena<T> Realize(T* ptr) const {
      std::vector<typename Arena<T>::Entry> map;
      for (const auto& entry : layout_) {
        map.push_back({ entry.name, TensorView<T>(ptr, entry.shape) });
        ptr += entry.shape.num_elements();
      }
      return Arena<T>(map);
    }

  private:
    std::vector<Entry> layout_;
};

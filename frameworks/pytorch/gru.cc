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

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

using haste::v0::gru::ForwardPass;
using haste::v0::gru::BackwardPass;

using torch::Tensor;

std::vector<Tensor> gru_forward(
    bool training,
    float zoneout_prob,
    Tensor x,
    Tensor h0,
    Tensor kernel,
    Tensor recurrent_kernel,
    Tensor bias,
    Tensor recurrent_bias,
    Tensor zoneout_mask) {
  const auto time_steps = x.size(0);
  const auto batch_size = x.size(1);
  const auto input_size = x.size(2);
  const auto hidden_size = recurrent_kernel.size(0);
  const bool has_zoneout = zoneout_prob && zoneout_mask.size(0);

  CHECK_INPUT(x);
  CHECK_INPUT(h0);
  CHECK_INPUT(kernel);
  CHECK_INPUT(recurrent_kernel);
  CHECK_INPUT(bias);
  CHECK_INPUT(recurrent_bias);
  CHECK_INPUT(zoneout_mask);

  Tensor output = torch::empty({ time_steps + 1, batch_size, hidden_size }, at::kCUDA);
  Tensor cache = torch::empty({ time_steps, batch_size, hidden_size * 4 }, at::kCUDA);
  Tensor tmp_Wx = torch::empty({ time_steps, batch_size, hidden_size * 3 }, at::kCUDA);
  Tensor tmp_Rh = torch::empty({ batch_size, hidden_size * 3 }, at::kCUDA);

  output[0] = h0;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gru_forward", ([&] {
    ForwardPass<scalar_t> forward(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle());

    auto x_a = x.packed_accessor32<scalar_t, 3>();
    auto output_a = output.packed_accessor32<scalar_t, 3>();
    auto cache_a = cache.packed_accessor32<scalar_t, 3>();
    auto tmp_Wx_a = tmp_Wx.packed_accessor32<scalar_t, 3>();
    auto zoneout_mask_a = zoneout_mask.packed_accessor32<scalar_t, 3>();

    for (auto i = decltype(time_steps){0}; i < time_steps; ++i) {
      forward.Iterate(
          kernel.data_ptr<scalar_t>(),
          recurrent_kernel.data_ptr<scalar_t>(),
          bias.data_ptr<scalar_t>(),
          recurrent_bias.data_ptr<scalar_t>(),
          x_a[i].data(),
          output_a[i].data(),
          output_a[i+1].data(),
          cache_a[i].data(),
          tmp_Wx_a[i].data(),
          tmp_Rh.data_ptr<scalar_t>(),
          has_zoneout ? zoneout_prob : 0.0f,
          has_zoneout ? zoneout_mask_a[i].data() : nullptr);
    }
  }));

  return { output, cache };
}

std::vector<Tensor> gru_backward(
    Tensor x_t,
    Tensor kernel_t,
    Tensor recurrent_kernel_t,
    Tensor bias,
    Tensor recurrent_bias,
    Tensor zoneout_mask,
    Tensor h_t,
    Tensor cache,
    Tensor dh_new) {
  const auto time_steps = x_t.size(0);
  const auto input_size = x_t.size(1);
  const auto batch_size = x_t.size(2);
  const auto hidden_size = recurrent_kernel_t.size(1);
  const bool has_zoneout = !!zoneout_mask.size(0);

  CHECK_INPUT(x_t);
  CHECK_INPUT(kernel_t);
  CHECK_INPUT(recurrent_kernel_t);
  CHECK_INPUT(bias);
  CHECK_INPUT(recurrent_bias);
  CHECK_INPUT(h_t);
  CHECK_INPUT(cache);
  CHECK_INPUT(dh_new);
  CHECK_INPUT(zoneout_mask);

  Tensor dx = torch::empty({ time_steps, batch_size, input_size }, at::kCUDA);
  Tensor dW = torch::zeros({ input_size, hidden_size * 3 }, at::kCUDA);
  Tensor dR = torch::zeros({ hidden_size, hidden_size * 3 }, at::kCUDA);
  Tensor dbx = torch::zeros({ hidden_size * 3 }, at::kCUDA);
  Tensor dbr = torch::zeros({ hidden_size * 3 }, at::kCUDA);
  Tensor dh = torch::zeros({ batch_size, hidden_size }, at::kCUDA);
  Tensor dp = torch::empty({ time_steps, batch_size, hidden_size * 3 }, at::kCUDA);
  Tensor dq = torch::empty({ time_steps, batch_size, hidden_size * 3 }, at::kCUDA);
  Tensor zeros = torch::zeros({ batch_size, hidden_size }, at::kCUDA);

  AT_DISPATCH_FLOATING_TYPES(x_t.scalar_type(), "gru_backward", ([&] {
    BackwardPass<scalar_t> backward(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle());

    auto x_t_a = x_t.packed_accessor32<scalar_t, 3>();
    auto h_t_a = h_t.packed_accessor32<scalar_t, 3>();
    auto cache_a = cache.packed_accessor32<scalar_t, 3>();
    auto dh_new_a = dh_new.packed_accessor32<scalar_t, 3>();
    auto dx_a = dx.packed_accessor32<scalar_t, 3>();
    auto dp_a = dp.packed_accessor32<scalar_t, 3>();
    auto dq_a = dq.packed_accessor32<scalar_t, 3>();
    auto zoneout_mask_a = zoneout_mask.packed_accessor32<scalar_t, 3>();

    for (auto i = time_steps - 1; i >= 0; --i) {
      backward.Iterate(
          kernel_t.data_ptr<scalar_t>(),
          recurrent_kernel_t.data_ptr<scalar_t>(),
          bias.data_ptr<scalar_t>(),
          recurrent_bias.data_ptr<scalar_t>(),
          x_t_a[i].data(),
          h_t_a[i].data(),
          cache_a[i].data(),
          dh_new_a[i+1].data(),
          dx_a[i].data(),
          dW.data_ptr<scalar_t>(),
          dR.data_ptr<scalar_t>(),
          dbx.data_ptr<scalar_t>(),
          dbr.data_ptr<scalar_t>(),
          dh.data_ptr<scalar_t>(),
          dp_a[i].data(),
          dq_a[i].data(),
          has_zoneout ? zoneout_mask_a[i].data() : nullptr);
    }
  }));

  return { dx, dh, dW, dR, dbx, dbr };
}

}  // anonymous namespace

void gru_init(py::module& m) {
  m.def("gru_forward", &gru_forward, "GRU forward");
  m.def("gru_backward", &gru_backward, "GRU backward");
}

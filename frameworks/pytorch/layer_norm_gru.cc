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
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "haste.h"
#include "support.h"

namespace {

namespace layer_norm = haste::v0::layer_norm;
namespace layer_norm_gru = haste::v0::layer_norm_gru;

using torch::Tensor;

std::vector<Tensor> layer_norm_gru_forward(
    bool training,
    float zoneout_prob,
    Tensor x,
    Tensor h0,
    Tensor kernel,
    Tensor recurrent_kernel,
    Tensor bias,
    Tensor recurrent_bias,
    Tensor gamma,
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
  CHECK_INPUT(gamma);
  CHECK_INPUT(zoneout_mask);

  const auto options = x.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  Tensor output = torch::empty({ time_steps + 1, batch_size, hidden_size }, options);
  Tensor cache = torch::empty({ time_steps, batch_size, hidden_size * 4 }, options);
  Tensor act_Wx = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
  Tensor tmp_Wx_norm = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
  Tensor act_Wx_norm_cache = torch::empty({ time_steps, batch_size, 2 }, options);
  Tensor act_Rh = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
  Tensor tmp_Rh_norm = torch::empty({ batch_size, hidden_size * 3 }, options);
  Tensor act_Rh_norm_cache = torch::empty({ time_steps, batch_size, 2 }, options);

  output[0] = h0;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layer_norm_gru_forward", ([&] {
    auto gamma_a = gamma.packed_accessor32<scalar_t, 2>();

    layer_norm::ForwardPass<scalar_t> layer_norm1(
        time_steps * batch_size,
        hidden_size * 3,
        gamma_a[0].data(),
        nullptr,
        act_Wx_norm_cache.data_ptr<scalar_t>());

    layer_norm::ForwardPass<scalar_t> layer_norm2(
        time_steps * batch_size,
        hidden_size * 3,
        gamma_a[1].data(),
        nullptr,
        act_Rh_norm_cache.data_ptr<scalar_t>());

    layer_norm_gru::ForwardPass<scalar_t> gru(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    gru.Run(
        time_steps,
        kernel.data_ptr<scalar_t>(),
        recurrent_kernel.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        recurrent_bias.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        cache.data_ptr<scalar_t>(),
        act_Wx.data_ptr<scalar_t>(),
        layer_norm1,
        tmp_Wx_norm.data_ptr<scalar_t>(),
        act_Rh.data_ptr<scalar_t>(),
        layer_norm2,
        tmp_Rh_norm.data_ptr<scalar_t>(),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? zoneout_mask.data_ptr<scalar_t>() : nullptr);
  }));

  return { output, cache, act_Wx, act_Wx_norm_cache, act_Rh, act_Rh_norm_cache };
}

std::vector<Tensor> layer_norm_gru_backward(
    Tensor x_t,
    Tensor kernel_t,
    Tensor recurrent_kernel_t,
    Tensor bias,
    Tensor recurrent_bias,
    Tensor gamma,
    Tensor zoneout_mask,
    Tensor h,
    Tensor cache,
    Tensor act_Wx,
    Tensor act_Wx_norm_cache,
    Tensor act_Rh,
    Tensor act_Rh_norm_cache,
    Tensor dh_new) {
  const auto input_size = x_t.size(0);
  const auto time_steps = x_t.size(1);
  const auto batch_size = x_t.size(2);
  const auto hidden_size = recurrent_kernel_t.size(1);
  const bool has_zoneout = !!zoneout_mask.size(0);

  CHECK_INPUT(x_t);
  CHECK_INPUT(kernel_t);
  CHECK_INPUT(recurrent_kernel_t);
  CHECK_INPUT(bias);
  CHECK_INPUT(recurrent_bias);
  CHECK_INPUT(gamma);
  CHECK_INPUT(h);
  CHECK_INPUT(cache);
  CHECK_INPUT(act_Wx);
  CHECK_INPUT(act_Wx_norm_cache);
  CHECK_INPUT(act_Rh);
  CHECK_INPUT(act_Rh_norm_cache);
  CHECK_INPUT(dh_new);
  CHECK_INPUT(zoneout_mask);

  const auto options = x_t.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
  Tensor dW = torch::zeros({ input_size, hidden_size * 3 }, options);
  Tensor dR = torch::zeros({ hidden_size, hidden_size * 3 }, options);
  Tensor dbx = torch::zeros({ hidden_size * 3 }, options);
  Tensor dbr = torch::zeros({ hidden_size * 3 }, options);
  Tensor dh = torch::zeros({ batch_size, hidden_size }, options);
  Tensor dp = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
  Tensor dq = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
  Tensor dgamma = torch::zeros_like(gamma);

  AT_DISPATCH_FLOATING_TYPES(x_t.scalar_type(), "layer_norm_gru_backward", ([&] {
    auto gamma_a = gamma.packed_accessor32<scalar_t, 2>();
    auto dgamma_a = dgamma.packed_accessor32<scalar_t, 2>();

    layer_norm::BackwardPass<scalar_t> layer_norm1(
        time_steps * batch_size,
        hidden_size * 3,
        gamma_a[0].data(),
        nullptr,
        act_Wx.data_ptr<scalar_t>(),
        dgamma_a[0].data(),
        nullptr,
        act_Wx_norm_cache.data_ptr<scalar_t>());

    layer_norm::BackwardPass<scalar_t> layer_norm2(
        time_steps * batch_size,
        hidden_size * 3,
        gamma_a[1].data(),
        nullptr,
        act_Rh.data_ptr<scalar_t>(),
        dgamma_a[1].data(),
        nullptr,
        act_Rh_norm_cache.data_ptr<scalar_t>());

    layer_norm_gru::BackwardPass<scalar_t> gru(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    gru.Run(
        time_steps,
        kernel_t.data_ptr<scalar_t>(),
        recurrent_kernel_t.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        recurrent_bias.data_ptr<scalar_t>(),
        x_t.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        cache.data_ptr<scalar_t>(),
        dh_new.data_ptr<scalar_t>(),
        dx.data_ptr<scalar_t>(),
        dW.data_ptr<scalar_t>(),
        dR.data_ptr<scalar_t>(),
        dbx.data_ptr<scalar_t>(),
        dbr.data_ptr<scalar_t>(),
        dh.data_ptr<scalar_t>(),
        dp.data_ptr<scalar_t>(),
        dq.data_ptr<scalar_t>(),
        layer_norm1,
        layer_norm2,
        has_zoneout ? zoneout_mask.data_ptr<scalar_t>() : nullptr);
  }));

  return { dx, dh, dW, dR, dbx, dbr, dgamma };
}

}  // anonymous namespace

void layer_norm_gru_init(py::module& m) {
  m.def("layer_norm_gru_forward", &layer_norm_gru_forward, "LayerNormGRU forward", py::call_guard<py::gil_scoped_release>());
  m.def("layer_norm_gru_backward", &layer_norm_gru_backward, "LayerNormGRU backward", py::call_guard<py::gil_scoped_release>());
}

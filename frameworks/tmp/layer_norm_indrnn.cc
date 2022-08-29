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
namespace layer_norm_indrnn = haste::v0::layer_norm_indrnn;

using torch::Tensor;

std::vector<Tensor> layer_norm_indrnn_forward(
    bool training,
    float zoneout_prob,
    Tensor x,
    Tensor h0,
    Tensor kernel,
    Tensor recurrent_scale,
    Tensor bias,
    Tensor gamma,
    Tensor zoneout_mask) {
  const auto time_steps = x.size(0);
  const auto batch_size = x.size(1);
  const auto input_size = x.size(2);
  const auto hidden_size = recurrent_scale.size(0);
  const bool has_zoneout = zoneout_prob && zoneout_mask.size(0);

  CHECK_INPUT(x);
  CHECK_INPUT(h0);
  CHECK_INPUT(kernel);
  CHECK_INPUT(recurrent_scale);
  CHECK_INPUT(bias);
  CHECK_INPUT(gamma);
  CHECK_INPUT(zoneout_mask);

  const auto options = x.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  Tensor output = torch::empty({ time_steps + 1, batch_size, hidden_size }, options);
  Tensor workspace = torch::empty({ time_steps, batch_size, hidden_size }, options);
  Tensor act_Wx = torch::empty({ time_steps, batch_size, hidden_size }, options);
  Tensor act_Wx_norm_cache = torch::empty({ time_steps, batch_size, 2 }, options);

  output[0] = h0;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layer_norm_indrnn_forward", ([&] {
    auto gamma_a = gamma.packed_accessor32<scalar_t, 2>();

    layer_norm::ForwardPass<scalar_t> layer_norm1(
        time_steps * batch_size,
        hidden_size,
        gamma_a[0].data(),
        nullptr,
        act_Wx_norm_cache.data_ptr<scalar_t>());

    layer_norm_indrnn::ForwardPass<scalar_t> forward(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        kernel.data_ptr<scalar_t>(),
        recurrent_scale.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        workspace.data_ptr<scalar_t>(),
        act_Wx.data_ptr<scalar_t>(),
        layer_norm1,
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? zoneout_mask.data_ptr<scalar_t>() : nullptr);
  }));

  return { output, act_Wx, act_Wx_norm_cache };
}

std::vector<Tensor> layer_norm_indrnn_backward(
    Tensor x_t,
    Tensor kernel_t,
    Tensor recurrent_scale,
    Tensor bias,
    Tensor gamma,
    Tensor zoneout_mask,
    Tensor h,
    Tensor act_Wx,
    Tensor act_Wx_norm_cache,
    Tensor dh_new) {
  const auto input_size = x_t.size(0);
  const auto time_steps = x_t.size(1);
  const auto batch_size = x_t.size(2);
  const auto hidden_size = recurrent_scale.size(0);
  const bool has_zoneout = !!zoneout_mask.size(0);

  CHECK_INPUT(x_t);
  CHECK_INPUT(kernel_t);
  CHECK_INPUT(recurrent_scale);
  CHECK_INPUT(bias);
  CHECK_INPUT(gamma);
  CHECK_INPUT(zoneout_mask);
  CHECK_INPUT(h);
  CHECK_INPUT(dh_new);

  const auto options = x_t.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
  Tensor dW = torch::zeros({ input_size, hidden_size }, options);
  Tensor du = torch::zeros({ hidden_size }, options);
  Tensor db = torch::zeros_like(bias);
  Tensor dh = torch::zeros({ batch_size, hidden_size }, options);
  Tensor workspace = torch::empty({ time_steps, batch_size, hidden_size }, options);
  Tensor dgamma = torch::zeros_like(gamma);

  AT_DISPATCH_FLOATING_TYPES(x_t.scalar_type(), "layer_norm_indrnn_backward", ([&] {
    auto gamma_a = gamma.packed_accessor32<scalar_t, 2>();
    auto dgamma_a = dgamma.packed_accessor32<scalar_t, 2>();

    layer_norm::BackwardPass<scalar_t> layer_norm1(
        time_steps * batch_size,
        hidden_size,
        gamma_a[0].data(),
        nullptr,
        act_Wx.data_ptr<scalar_t>(),
        dgamma_a[0].data(),
        nullptr,
        act_Wx_norm_cache.data_ptr<scalar_t>());

    layer_norm_indrnn::BackwardPass<scalar_t> backward(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        kernel_t.data_ptr<scalar_t>(),
        recurrent_scale.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        x_t.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        dh_new.data_ptr<scalar_t>(),
        dx.data_ptr<scalar_t>(),
        dW.data_ptr<scalar_t>(),
        du.data_ptr<scalar_t>(),
        db.data_ptr<scalar_t>(),
        dh.data_ptr<scalar_t>(),
        workspace.data_ptr<scalar_t>(),
        layer_norm1,
        has_zoneout ? zoneout_mask.data_ptr<scalar_t>() : nullptr);
  }));

  return { dx, dh, dW, du, db, dgamma };
}

}  // anonymous namespace

void layer_norm_indrnn_init(py::module& m) {
  m.def("layer_norm_indrnn_forward", &layer_norm_indrnn_forward, "LayerNormIndRNN forward", py::call_guard<py::gil_scoped_release>());
  m.def("layer_norm_indrnn_backward", &layer_norm_indrnn_backward, "LayerNormIndRNN backward", py::call_guard<py::gil_scoped_release>());
}

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

namespace layer_norm = haste::v0::layer_norm;
namespace layer_norm_lstm = haste::v0::layer_norm_lstm;

using torch::Tensor;

std::vector<Tensor> layer_norm_lstm_forward(
    bool training,
    float zoneout_prob,
    Tensor x,
    Tensor h0,
    Tensor c0,
    Tensor kernel,
    Tensor recurrent_kernel,
    Tensor bias,
    Tensor gamma,
    Tensor gamma_h,
    Tensor beta_h,
    Tensor zoneout_mask) {
  const auto time_steps = x.size(0);
  const auto batch_size = x.size(1);
  const auto input_size = x.size(2);
  const auto hidden_size = recurrent_kernel.size(0);
  const bool has_zoneout = zoneout_prob && zoneout_mask.size(0);

  CHECK_INPUT(x);
  CHECK_INPUT(h0);
  CHECK_INPUT(c0);
  CHECK_INPUT(kernel);
  CHECK_INPUT(recurrent_kernel);
  CHECK_INPUT(bias);
  CHECK_INPUT(gamma);
  CHECK_INPUT(gamma_h);
  CHECK_INPUT(beta_h);
  CHECK_INPUT(zoneout_mask);

  Tensor output = torch::zeros({ time_steps + 1, batch_size, hidden_size }, at::kCUDA);
  Tensor output_state = torch::zeros({ time_steps + 1, batch_size, hidden_size }, at::kCUDA);
  Tensor act_Wx = torch::empty({ time_steps, batch_size, hidden_size * 4 }, at::kCUDA);
  Tensor act_Wx_norm = torch::empty({ time_steps, batch_size, hidden_size * 4 }, at::kCUDA);
  Tensor act_Wx_norm_cache = torch::empty({ time_steps, batch_size, 2 }, at::kCUDA);
  Tensor act_Rh = torch::empty({ time_steps, batch_size, hidden_size * 4 }, at::kCUDA);
  Tensor act_Rh_norm_cache = torch::empty({ time_steps, batch_size, 2 }, at::kCUDA);
  Tensor act_c_norm = torch::empty({ time_steps, batch_size, hidden_size }, at::kCUDA);
  Tensor act_c_norm_cache = torch::empty({ time_steps, batch_size, 2 }, at::kCUDA);
  Tensor tmp_Rh = torch::empty({ batch_size, hidden_size * 4 }, at::kCUDA);

  output[0] = h0;
  output_state[0] = c0;

  AT_DISPATCH_FLOATING_TYPES(x.type(), "layer_norm_lstm_forward", ([&] {
    auto gamma_a = gamma.packed_accessor<scalar_t, 2>();

    layer_norm::ForwardPass<scalar_t> layer_norm1(
        time_steps * batch_size,
        hidden_size * 4,
        gamma_a[0].data(),
        nullptr,
        act_Wx_norm_cache.data<scalar_t>());

    layer_norm::ForwardPass<scalar_t> layer_norm2(
        time_steps * batch_size,
        hidden_size * 4,
        gamma_a[1].data(),
        nullptr,
        act_Rh_norm_cache.data<scalar_t>());

    layer_norm::ForwardPass<scalar_t> layer_norm3(
        time_steps * batch_size,
        hidden_size,
        gamma_h.data<scalar_t>(),
        beta_h.data<scalar_t>(),
        act_c_norm_cache.data<scalar_t>());

    layer_norm_lstm::ForwardPass<scalar_t> lstm(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle());

    lstm.Run(
        time_steps,
        kernel.data<scalar_t>(),
        recurrent_kernel.data<scalar_t>(),
        bias.data<scalar_t>(),
        x.data<scalar_t>(),
        output.data<scalar_t>(),
        output_state.data<scalar_t>(),
        act_Wx.data<scalar_t>(),
        tmp_Rh.data<scalar_t>(),
        layer_norm1,
        act_Wx_norm.data<scalar_t>(),
        act_Rh.data<scalar_t>(),
        layer_norm2,
        layer_norm3,
        act_c_norm.data<scalar_t>(),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? zoneout_mask.data<scalar_t>() : nullptr);
  }));

  return {
      output,
      output_state,
      act_Wx,
      act_Wx_norm,
      act_Wx_norm_cache,
      act_Rh,
      act_Rh_norm_cache,
      act_c_norm,
      act_c_norm_cache };
}

std::vector<Tensor> layer_norm_lstm_backward(
    Tensor x_t,
    Tensor kernel_t,
    Tensor recurrent_kernel_t,
    Tensor bias,
    Tensor gamma,
    Tensor gamma_h,
    Tensor beta_h,
    Tensor zoneout_mask,
    Tensor h,
    Tensor c,
    Tensor act_Wx,
    Tensor act_Wx_norm,
    Tensor act_Wx_norm_cache,
    Tensor act_Rh,
    Tensor act_Rh_norm_cache,
    Tensor act_c_norm,
    Tensor act_c_norm_cache,
    Tensor dh_new,
    Tensor dc_new) {
  const auto input_size = x_t.size(0);
  const auto time_steps = x_t.size(1);
  const auto batch_size = x_t.size(2);
  const auto hidden_size = recurrent_kernel_t.size(1);
  const bool has_zoneout = !!zoneout_mask.size(0);

  CHECK_INPUT(x_t);
  CHECK_INPUT(kernel_t);
  CHECK_INPUT(recurrent_kernel_t);
  CHECK_INPUT(bias);
  CHECK_INPUT(gamma);
  CHECK_INPUT(gamma_h);
  CHECK_INPUT(beta_h);
  CHECK_INPUT(zoneout_mask);
  CHECK_INPUT(h);
  CHECK_INPUT(c);
  CHECK_INPUT(act_Wx);
  CHECK_INPUT(act_Wx_norm);
  CHECK_INPUT(act_Wx_norm_cache);
  CHECK_INPUT(act_Rh);
  CHECK_INPUT(act_Rh_norm_cache);
  CHECK_INPUT(act_c_norm);
  CHECK_INPUT(act_c_norm_cache);
  CHECK_INPUT(dh_new);
  CHECK_INPUT(dc_new);

  Tensor dx = torch::empty({ time_steps, batch_size, input_size }, at::kCUDA);
  Tensor dW = torch::zeros({ input_size, hidden_size * 4 }, at::kCUDA);
  Tensor dR = torch::zeros({ hidden_size, hidden_size * 4 }, at::kCUDA);
  Tensor db = torch::zeros({ hidden_size * 4 }, at::kCUDA);
  Tensor dgamma = torch::zeros_like(gamma);
  Tensor dgamma_h = torch::zeros_like(gamma_h);
  Tensor dbeta_h = torch::zeros_like(beta_h);
  Tensor dh = torch::zeros({ batch_size, hidden_size }, at::kCUDA);
  Tensor dc = torch::zeros({ batch_size, hidden_size }, at::kCUDA);

  AT_DISPATCH_FLOATING_TYPES(x_t.type(), "layer_norm_lstm_backward", ([&] {
    auto gamma_a = gamma.packed_accessor<scalar_t, 2>();
    auto dgamma_a = dgamma.packed_accessor<scalar_t, 2>();
    auto c_a = c.packed_accessor<scalar_t, 3>();

    layer_norm::BackwardPass<scalar_t> layer_norm1(
        time_steps * batch_size,
        hidden_size * 4,
        gamma_a[0].data(),
        nullptr,
        act_Wx.data<scalar_t>(),
        dgamma_a[0].data(),
        nullptr,
        act_Wx_norm_cache.data<scalar_t>());

    layer_norm::BackwardPass<scalar_t> layer_norm2(
        time_steps * batch_size,
        hidden_size * 4,
        gamma_a[1].data(),
        nullptr,
        act_Rh.data<scalar_t>(),
        dgamma_a[1].data(),
        nullptr,
        act_Rh_norm_cache.data<scalar_t>());

    layer_norm::BackwardPass<scalar_t> layer_norm3(
        time_steps * batch_size,
        hidden_size,
        gamma_h.data<scalar_t>(),
        beta_h.data<scalar_t>(),
        c_a[1].data(),
        dgamma_h.data<scalar_t>(),
        dbeta_h.data<scalar_t>(),
        act_c_norm_cache.data<scalar_t>());

    layer_norm_lstm::BackwardPass<scalar_t> lstm(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle());

    lstm.Run(
        time_steps,
        kernel_t.data<scalar_t>(),
        recurrent_kernel_t.data<scalar_t>(),
        bias.data<scalar_t>(),
        x_t.data<scalar_t>(),
        h.data<scalar_t>(),
        c.data<scalar_t>(),
        dh_new.data<scalar_t>(),
        dc_new.data<scalar_t>(),
        dx.data<scalar_t>(),
        dW.data<scalar_t>(),
        dR.data<scalar_t>(),
        db.data<scalar_t>(),
        dh.data<scalar_t>(),
        dc.data<scalar_t>(),
        act_Wx.data<scalar_t>(),
        layer_norm1,
        act_Wx_norm.data<scalar_t>(),
        act_Rh.data<scalar_t>(),
        layer_norm2,
        layer_norm3,
        act_c_norm.data<scalar_t>(),
        has_zoneout ? zoneout_mask.data<scalar_t>() : nullptr);
  }));

  return { dx, dh, dc, dW, dR, db, dgamma, dgamma_h, dbeta_h };
}

}  // anonymous namespace

void layer_norm_lstm_init(py::module& m) {
  m.def("layer_norm_lstm_forward", &layer_norm_lstm_forward, "LayerNormLSTM forward");
  m.def("layer_norm_lstm_backward", &layer_norm_lstm_backward, "LayerNormLSTM backward");
}

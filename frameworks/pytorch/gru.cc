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
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        kernel.data_ptr<scalar_t>(),
        recurrent_kernel.data_ptr<scalar_t>(),
        bias.data_ptr<scalar_t>(),
        recurrent_bias.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        cache.data_ptr<scalar_t>(),
        tmp_Wx.data_ptr<scalar_t>(),
        tmp_Rh.data_ptr<scalar_t>(),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? zoneout_mask.data_ptr<scalar_t>() : nullptr);
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
    Tensor h,
    Tensor cache,
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
  CHECK_INPUT(h);
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

  AT_DISPATCH_FLOATING_TYPES(x_t.scalar_type(), "gru_backward", ([&] {
    BackwardPass<scalar_t> backward(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
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
        has_zoneout ? zoneout_mask.data_ptr<scalar_t>() : nullptr);
  }));

  return { dx, dh, dW, dR, dbx, dbr };
}

}  // anonymous namespace

void gru_init(py::module& m) {
  m.def("gru_forward", &gru_forward, "GRU forward");
  m.def("gru_backward", &gru_backward, "GRU backward");
}

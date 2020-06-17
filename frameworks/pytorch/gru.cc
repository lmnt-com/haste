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

  const auto options = x.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  Tensor output = torch::empty({ time_steps + 1, batch_size, hidden_size }, options);
  Tensor cache = torch::empty({ time_steps, batch_size, hidden_size * 4 }, options);
  Tensor tmp_Wx = torch::empty({ time_steps, batch_size, hidden_size * 3 }, options);
  Tensor tmp_Rh = torch::empty({ batch_size, hidden_size * 3 }, options);

  output[0] = h0;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "gru_forward", ([&] {
    ForwardPass<typename native_type<scalar_t>::T> forward(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        time_steps,
        ptr<scalar_t>(kernel),
        ptr<scalar_t>(recurrent_kernel),
        ptr<scalar_t>(bias),
        ptr<scalar_t>(recurrent_bias),
        ptr<scalar_t>(x),
        ptr<scalar_t>(output),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(tmp_Wx),
        ptr<scalar_t>(tmp_Rh),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr);
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_t.scalar_type(), "gru_backward", ([&] {
    BackwardPass<typename native_type<scalar_t>::T> backward(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    backward.Run(
        time_steps,
        ptr<scalar_t>(kernel_t),
        ptr<scalar_t>(recurrent_kernel_t),
        ptr<scalar_t>(bias),
        ptr<scalar_t>(recurrent_bias),
        ptr<scalar_t>(x_t),
        ptr<scalar_t>(h),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(dh_new),
        ptr<scalar_t>(dx),
        ptr<scalar_t>(dW),
        ptr<scalar_t>(dR),
        ptr<scalar_t>(dbx),
        ptr<scalar_t>(dbr),
        ptr<scalar_t>(dh),
        ptr<scalar_t>(dp),
        ptr<scalar_t>(dq),
        has_zoneout ? ptr<scalar_t>(zoneout_mask) : nullptr);
  }));

  return { dx, dh, dW, dR, dbx, dbr };
}

}  // anonymous namespace

void gru_init(py::module& m) {
  m.def("gru_forward", &gru_forward, "GRU forward", py::call_guard<py::gil_scoped_release>());
  m.def("gru_backward", &gru_backward, "GRU backward", py::call_guard<py::gil_scoped_release>());
}

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

using haste::v0::lstm::ForwardPass;
using haste::v0::lstm::BackwardPass;

using torch::Tensor;

std::vector<Tensor> lstm_forward(
    bool training,
    float zoneout_prob,
    Tensor x,
    Tensor kernel,
    Tensor recurrent_kernel,
    Tensor bias,
    Tensor zoneout_mask) {
  const auto time_steps = x.size(0);
  const auto batch_size = x.size(1);
  const auto input_size = x.size(2);
  const auto hidden_size = recurrent_kernel.size(0);
  const bool has_zoneout = zoneout_prob && zoneout_mask.size(0);

  CHECK_INPUT(x);
  CHECK_INPUT(kernel);
  CHECK_INPUT(recurrent_kernel);
  CHECK_INPUT(bias);
  CHECK_INPUT(zoneout_mask);

  Tensor output = torch::zeros({ time_steps + 1, batch_size, hidden_size }, at::kCUDA);
  Tensor output_state = torch::zeros({ time_steps + 1, batch_size, hidden_size }, at::kCUDA);
  Tensor cache = torch::empty({ time_steps, batch_size, hidden_size * 4 }, at::kCUDA);
  Tensor tmp_Rh = torch::empty({ batch_size, hidden_size * 4 }, at::kCUDA);

  AT_DISPATCH_FLOATING_TYPES(x.type(), "lstm_forward", ([&] {
    ForwardPass<scalar_t> forward(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle());

    forward.Run(
        time_steps,
        kernel.data<scalar_t>(),
        recurrent_kernel.data<scalar_t>(),
        bias.data<scalar_t>(),
        x.data<scalar_t>(),
        output.data<scalar_t>(),
        output_state.data<scalar_t>(),
        cache.data<scalar_t>(),
        tmp_Rh.data<scalar_t>(),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? zoneout_mask.data<scalar_t>() : nullptr);
  }));

  return { output, output_state, cache };
}

std::vector<Tensor> lstm_backward(
    Tensor x_t,
    Tensor kernel_t,
    Tensor recurrent_kernel_t,
    Tensor bias,
    Tensor zoneout_mask,
    Tensor h,
    Tensor c,
    Tensor cache,
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
  CHECK_INPUT(h);
  CHECK_INPUT(c);
  CHECK_INPUT(cache);
  CHECK_INPUT(dh_new);
  CHECK_INPUT(dc_new);
  CHECK_INPUT(zoneout_mask);

  Tensor dx = torch::empty({ time_steps, batch_size, input_size }, at::kCUDA);
  Tensor dW = torch::zeros({ input_size, hidden_size * 4 }, at::kCUDA);
  Tensor dR = torch::zeros({ hidden_size, hidden_size * 4 }, at::kCUDA);
  Tensor db = torch::zeros_like(bias);
  Tensor dh = torch::zeros({ batch_size, hidden_size }, at::kCUDA);
  Tensor dc = torch::zeros({ batch_size, hidden_size }, at::kCUDA);

  AT_DISPATCH_FLOATING_TYPES(x_t.type(), "lstm_backward", ([&] {
    BackwardPass<scalar_t> backward(
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle());

    backward.Run(
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
        cache.data<scalar_t>(),
        has_zoneout ? zoneout_mask.data<scalar_t>() : nullptr);
  }));

  return { dx, dW, dR, db };
}

}  // anonymous namespace

void lstm_init(py::module& m) {
  m.def("lstm_forward", &lstm_forward, "LSTM forward");
  m.def("lstm_backward", &lstm_backward, "LSTM backward");
}

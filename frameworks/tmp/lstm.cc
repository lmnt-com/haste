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

using haste::v0::lstm::ForwardPass;
using haste::v0::lstm::BackwardPass;

using torch::Tensor;

std::vector<Tensor> lstm_forward(
    bool training,
    float zoneout_prob,
    Tensor x,
    Tensor h0,
    Tensor c0,
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
  CHECK_INPUT(h0);
  CHECK_INPUT(c0);
  CHECK_INPUT(kernel);
  CHECK_INPUT(recurrent_kernel);
  CHECK_INPUT(bias);
  CHECK_INPUT(zoneout_mask);

  const auto options = x.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  Tensor output = torch::empty({ time_steps + 1, batch_size, hidden_size }, options);
  Tensor output_state = torch::empty({ time_steps + 1, batch_size, hidden_size }, options);
  Tensor cache = torch::empty({ time_steps, batch_size, hidden_size * 4 }, options);
  Tensor tmp_Rh = torch::empty({ batch_size, hidden_size * 4 }, options);

  output[0] = h0;
  output_state[0] = c0;

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lstm_forward", ([&] {
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
        x.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        output_state.data_ptr<scalar_t>(),
        cache.data_ptr<scalar_t>(),
        tmp_Rh.data_ptr<scalar_t>(),
        has_zoneout ? zoneout_prob : 0.0f,
        has_zoneout ? zoneout_mask.data_ptr<scalar_t>() : nullptr);
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

  const auto options = x_t.options();
  const at::cuda::CUDAGuard guard(options.device_index());
  Tensor dx = torch::empty({ time_steps, batch_size, input_size }, options);
  Tensor dW = torch::zeros({ input_size, hidden_size * 4 }, options);
  Tensor dR = torch::zeros({ hidden_size, hidden_size * 4 }, options);
  Tensor db = torch::zeros_like(bias);
  Tensor dh = torch::zeros({ batch_size, hidden_size }, options);
  Tensor dc = torch::zeros({ batch_size, hidden_size }, options);

  AT_DISPATCH_FLOATING_TYPES(x_t.scalar_type(), "lstm_backward", ([&] {
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
        x_t.data_ptr<scalar_t>(),
        h.data_ptr<scalar_t>(),
        c.data_ptr<scalar_t>(),
        dh_new.data_ptr<scalar_t>(),
        dc_new.data_ptr<scalar_t>(),
        dx.data_ptr<scalar_t>(),
        dW.data_ptr<scalar_t>(),
        dR.data_ptr<scalar_t>(),
        db.data_ptr<scalar_t>(),
        dh.data_ptr<scalar_t>(),
        dc.data_ptr<scalar_t>(),
        cache.data_ptr<scalar_t>(),
        has_zoneout ? zoneout_mask.data_ptr<scalar_t>() : nullptr);
  }));

  return { dx, dh, dc, dW, dR, db };
}

}  // anonymous namespace

void lstm_init(py::module& m) {
  m.def("lstm_forward", &lstm_forward, "LSTM forward", py::call_guard<py::gil_scoped_release>());
  m.def("lstm_backward", &lstm_backward, "LSTM backward", py::call_guard<py::gil_scoped_release>());
}

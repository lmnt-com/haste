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

#include "ligru_1_0.h"
#include "support.h"

namespace {

using haste::v0::ligru_1_0::BackwardPass;
using haste::v0::ligru_1_0::ForwardPass;

using torch::Tensor;

std::vector<Tensor> ligru_1_0_forward(bool training, Tensor wx, Tensor h_init,
                                  Tensor u_t) {

  const auto seq_length = wx.size(0);
  const auto batch_size = wx.size(1);
  const auto hidden_size = h_init.size(1);

  CHECK_INPUT(wx);
  CHECK_INPUT(h_init);
  CHECK_INPUT(u_t);

  const auto options = wx.options();
  const at::cuda::CUDAGuard guard(options.device_index());

  Tensor output =
      torch::empty({seq_length + 1, batch_size, hidden_size}, options);
  Tensor cache =
      torch::empty({seq_length, batch_size, hidden_size * 3}, options);
  Tensor tmp_uh = torch::zeros({batch_size, hidden_size * 2}, options);

  output[0] = h_init;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      wx.scalar_type(), "ligru_forward", ([&] {
        ForwardPass<typename native_type<scalar_t>::T> forward(
            training, batch_size, 0, hidden_size,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        forward.Run(seq_length, ptr<scalar_t>(wx), ptr<scalar_t>(u_t),
                    ptr<scalar_t>(output), ptr<scalar_t>(cache),
                    ptr<scalar_t>(tmp_uh));
      }));

  return {output, cache};
}

std::vector<Tensor> ligru_1_0_backward(Tensor wx, Tensor u, Tensor h,
                                   Tensor cache, Tensor grad_out) {

  const auto input_size = wx.size(0);
  const auto time_steps = wx.size(0);
  const auto batch_size = wx.size(1);
  const auto hidden_size = wx.size(2) / 2;

  CHECK_INPUT(wx);
  CHECK_INPUT(u);
  CHECK_INPUT(h);
  CHECK_INPUT(cache);
  CHECK_INPUT(grad_out);

  const auto options = wx.options();
  const at::cuda::CUDAGuard guard(options.device_index());

  Tensor dwx = torch::zeros({time_steps, batch_size, hidden_size * 2}, options);
  Tensor du = torch::zeros({hidden_size, hidden_size * 2}, options);
  Tensor dh = torch::zeros({batch_size, hidden_size}, options);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      wx.scalar_type(), "ligru_backward", ([&] {
        BackwardPass<typename native_type<scalar_t>::T> backward(
            batch_size, input_size, hidden_size,
            at::cuda::getCurrentCUDABlasHandle(),
            at::cuda::getCurrentCUDAStream());

        backward.Run(time_steps, ptr<scalar_t>(wx), ptr<scalar_t>(u),
                     ptr<scalar_t>(h), ptr<scalar_t>(cache),
                     ptr<scalar_t>(grad_out), ptr<scalar_t>(dwx),
                     ptr<scalar_t>(du), ptr<scalar_t>(dh));
      }));

  return {du, dwx, dh};
}

} // anonymous namespace

void ligru_1_0_init(py::module &m) {
  m.def("ligru_1_0_forward", &ligru_1_0_forward, "Li-GRU forward",
        py::call_guard<py::gil_scoped_release>());
  m.def("ligru_1_0_backward", &ligru_1_0_backward, "Li-GRU backward",
        py::call_guard<py::gil_scoped_release>());
}

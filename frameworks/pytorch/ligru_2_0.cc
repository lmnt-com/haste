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
namespace layer_norm_ligru = haste::v0::ligru_v2;

// namespace layer_norm_ligru_2_0 = haste::v0::ligru_v2;
// namespace layer_norm = haste::v0::layer_norm; 

using torch::Tensor;

std::vector<Tensor> ligru_forward(
    bool training,
    Tensor wx, 
    Tensor h_init, 
    Tensor u,
    Tensor drop_mask
) {

  const auto seq_length = wx.size(0);
  const auto batch_size = wx.size(1);
  // const auto input_size = x.size(2);
  const auto hidden_size = h_init.size(1);

  CHECK_INPUT(wx);
  CHECK_INPUT(h_init);
  CHECK_INPUT(u);
  CHECK_INPUT(drop_mask);

  const auto options = wx.options();
  const at::cuda::CUDAGuard guard(options.device_index());

  Tensor output = torch::empty({ seq_length + 1, batch_size, hidden_size }, options);
  Tensor cache = torch::empty({ seq_length, batch_size, hidden_size * 3 }, options);

  Tensor act_uh = torch::zeros({ seq_length, batch_size, hidden_size * 2 }, options);
  Tensor tmp_uh_norm = torch::zeros({ batch_size, hidden_size * 2 }, options);
  Tensor act_uh_norm_cache = torch::zeros({ seq_length, batch_size, 2 }, options);

  Tensor gamma = torch::ones({2, hidden_size}, options);

  output[0] = h_init; 

  AT_DISPATCH_FLOATING_TYPES(wx.scalar_type(), "ligru_layer_norm_forward", ([&] {
    auto gamma_a = gamma.packed_accessor32<scalar_t, 2>();
    layer_norm::ForwardPass<scalar_t> layer_norm1(
        seq_length * batch_size,
        hidden_size * 2,
        gamma_a[0].data(),
        nullptr,
        act_uh_norm_cache.data_ptr<scalar_t>());    


    layer_norm_ligru::ForwardPass<typename native_type<scalar_t>::T> forward(
        training,
        batch_size,
        0,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());

    forward.Run(
        seq_length,
        wx.data_ptr<scalar_t>(),
        u.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        cache.data_ptr<scalar_t>(),
        layer_norm1,
        tmp_uh_norm.data_ptr<scalar_t>(),
        act_uh.data_ptr<scalar_t>(),
        drop_mask.data_ptr<scalar_t>());

    

  }));

  return {output, cache, act_uh, act_uh_norm_cache};
}

std::vector<Tensor> ligru_backward(
    Tensor wx_t,
    Tensor u_t,
    Tensor drop_mask,
    Tensor h,
    Tensor cache,
    Tensor act_uh, 
    Tensor act_uh_norm_cache,
    Tensor grad_out) {

    const auto input_size = wx_t.size(0);
    const auto time_steps = wx_t.size(0);
    const auto batch_size = wx_t.size(1);
    const auto hidden_size = wx_t.size(2) / 2;

    CHECK_INPUT(wx_t);
    CHECK_INPUT(u_t);
    CHECK_INPUT(h);
    CHECK_INPUT(cache);
    CHECK_INPUT(grad_out);
    CHECK_INPUT(drop_mask);
    CHECK_INPUT(act_uh);
    CHECK_INPUT(act_uh_norm_cache);

    const auto options = wx_t.options();
    const at::cuda::CUDAGuard guard(options.device_index());

    Tensor dwx = torch::zeros({time_steps, batch_size, hidden_size * 2 }, options);
    Tensor tmp_dwx = torch::zeros({time_steps, batch_size, hidden_size * 2 }, options);
    Tensor du = torch::zeros({hidden_size, hidden_size * 2}, options);
    Tensor dh = torch::zeros({ batch_size, hidden_size }, options);
    Tensor gamma = torch::ones({2, hidden_size}, options);
    Tensor dgamma = torch::zeros_like(gamma);

    AT_DISPATCH_FLOATING_TYPES(wx_t.scalar_type(), "ligru_2_0_backward", ([&] {
        auto gamma_a = gamma.packed_accessor32<scalar_t, 2>();
        auto dgamma_a = dgamma.packed_accessor32<scalar_t, 2>();

        layer_norm::BackwardPass<scalar_t> layer_norm1(
          time_steps * batch_size,
          hidden_size * 2,
          gamma_a[0].data(),
          nullptr,
          act_uh.data_ptr<scalar_t>(),
          dgamma_a[0].data(),
          nullptr,
          act_uh_norm_cache.data_ptr<scalar_t>()
        );

        layer_norm_ligru::BackwardPass<scalar_t> backward(
          batch_size,
          input_size,
          hidden_size,
          at::cuda::getCurrentCUDABlasHandle(),
          at::cuda::getCurrentCUDAStream()
        );

        backward.Run(
          time_steps,
          wx_t.data_ptr<scalar_t>(),
          u_t.data_ptr<scalar_t>(),
          h.data_ptr<scalar_t>(),
          cache.data_ptr<scalar_t>(),
          grad_out.data_ptr<scalar_t>(),
          tmp_dwx.data_ptr<scalar_t>(),
          dwx.data_ptr<scalar_t>(),
          du.data_ptr<scalar_t>(),
          dh.data_ptr<scalar_t>(),
          layer_norm1,
          drop_mask.data_ptr<scalar_t>()
        );

    }));

    return {du, dwx, tmp_dwx};

  }
}  // anonymous namespace

void ligru_2_0_init(py::module& m) {
  m.def("ligru_2_0_forward", &ligru_forward, "Li-GRU 2.0 forward", py::call_guard<py::gil_scoped_release>());
  m.def("ligru_2_0_backward", &ligru_backward, "Li-GRU 2.0 backward", py::call_guard<py::gil_scoped_release>());

}

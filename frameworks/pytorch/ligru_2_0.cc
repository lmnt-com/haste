// Copyright 2022 Adel Moumen, All Rights Reserved.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "layer_norm.h"
#include "ligru_2_0.h"
#include "support.h"

namespace {

namespace layer_norm = haste::v0::layer_norm;
namespace layer_norm_ligru = haste::v0::ligru_2_0;

using torch::Tensor;

std::vector<Tensor> ligru_2_0_forward(bool training, Tensor wx, Tensor h_init,
                                  Tensor u_t, int activation) {

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

  Tensor act_uh =
      torch::empty({seq_length, batch_size, hidden_size * 2}, options);
  Tensor tmp_uh_norm = torch::empty({batch_size, hidden_size * 2}, options);
  Tensor act_uh_norm_cache = torch::empty({seq_length, batch_size, 2}, options);

  output[0] = h_init;

  AT_DISPATCH_FLOATING_TYPES(
      wx.scalar_type(), "ligru_2_0_forward", ([&] {
        layer_norm::ForwardPass<scalar_t> layer_norm1(
            seq_length * batch_size, hidden_size * 2, nullptr,
            nullptr, act_uh_norm_cache.data_ptr<scalar_t>());

        layer_norm_ligru::ForwardPass<typename native_type<scalar_t>::T>
            forward(training, batch_size, 0, hidden_size,
                    at::cuda::getCurrentCUDABlasHandle(),
                    activation,
                    at::cuda::getCurrentCUDAStream());

        forward.Run(seq_length, wx.data_ptr<scalar_t>(), u_t.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(), cache.data_ptr<scalar_t>(),
                    layer_norm1, tmp_uh_norm.data_ptr<scalar_t>(),
                    act_uh.data_ptr<scalar_t>());
      }));

  return {output, cache, act_uh, act_uh_norm_cache};
}

std::vector<Tensor> ligru_2_0_backward(Tensor wx, Tensor u, Tensor h,
                                   Tensor cache, Tensor act_uh,
                                   Tensor act_uh_norm_cache, Tensor grad_out, int activation) {

  const auto input_size = wx.size(0);
  const auto time_steps = wx.size(0);
  const auto batch_size = wx.size(1);
  const auto hidden_size = wx.size(2) / 2;

  CHECK_INPUT(wx);
  CHECK_INPUT(u);
  CHECK_INPUT(h);
  CHECK_INPUT(cache);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(act_uh);
  CHECK_INPUT(act_uh_norm_cache);

  const auto options = wx.options();
  const at::cuda::CUDAGuard guard(options.device_index());

  Tensor dwx = torch::empty({time_steps, batch_size, hidden_size * 2}, options);
  Tensor tmp_dwx =
      torch::empty({time_steps, batch_size, hidden_size * 2}, options);
  Tensor du = torch::zeros({hidden_size, hidden_size * 2}, options);
  Tensor dh = torch::zeros({batch_size, hidden_size}, options);

  AT_DISPATCH_FLOATING_TYPES(
      wx.scalar_type(), "ligru_2_0_backward", ([&] {

        layer_norm::BackwardPass<scalar_t> layer_norm1(
            time_steps * batch_size, hidden_size * 2, nullptr,
            nullptr, act_uh.data_ptr<scalar_t>(), nullptr, nullptr,
            act_uh_norm_cache.data_ptr<scalar_t>());

        layer_norm_ligru::BackwardPass<scalar_t> backward(
            batch_size, input_size, hidden_size,
            at::cuda::getCurrentCUDABlasHandle(),
            activation,
            at::cuda::getCurrentCUDAStream());

        backward.Run(time_steps, wx.data_ptr<scalar_t>(),
                     u.data_ptr<scalar_t>(), h.data_ptr<scalar_t>(),
                     cache.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
                     tmp_dwx.data_ptr<scalar_t>(), dwx.data_ptr<scalar_t>(),
                     du.data_ptr<scalar_t>(), dh.data_ptr<scalar_t>(),
                     layer_norm1);
      }));

  return {du, dwx, tmp_dwx};
}
} // anonymous namespace

void ligru_2_0_init(py::module &m) {
  m.def("ligru_2_0_forward", &ligru_2_0_forward, "Li-GRU 2.0 forward",
        py::call_guard<py::gil_scoped_release>());
  m.def("ligru_2_0_backward", &ligru_2_0_backward, "Li-GRU 2.0 backward",
        py::call_guard<py::gil_scoped_release>());
}

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

using haste::v0::ligru::ForwardPass;

using torch::Tensor;

std::vector<Tensor> ligru_forward(
    bool training,
    Tensor x,
    Tensor h_init, 
    Tensor w, 
    Tensor u,
    Tensor drop_mask
) {

  const auto seq_length = x.size(0);
  const auto batch_size = x.size(1);
  const auto input_size = x.size(2);
  const auto hidden_size = h_init.size(1);


  CHECK_INPUT(x);
  CHECK_INPUT(h_init);
  CHECK_INPUT(w);
  CHECK_INPUT(u);
  CHECK_INPUT(drop_mask);

  const auto options = x.options();
  const at::cuda::CUDAGuard guard(options.device_index());

  Tensor output = torch::empty({ seq_length + 1, batch_size, hidden_size }, options);
  Tensor cache = torch::empty({ seq_length, batch_size, hidden_size * 3 }, options);
  Tensor tmp_wx = torch::zeros({ seq_length, batch_size, hidden_size * 2 }, options);
  Tensor tmp_uh = torch::empty({ batch_size, hidden_size * 2 }, options);

  output[0] = h_init;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "ligru_forward", ([&] {
    ForwardPass<typename native_type<scalar_t>::T> forward(
        training,
        batch_size,
        input_size,
        hidden_size,
        at::cuda::getCurrentCUDABlasHandle(),
        at::cuda::getCurrentCUDAStream());


    forward.Run(
        seq_length,
        ptr<scalar_t>(w),
        ptr<scalar_t>(u),
        ptr<scalar_t>(x),
        ptr<scalar_t>(output),
        ptr<scalar_t>(cache),
        ptr<scalar_t>(tmp_wx),
        ptr<scalar_t>(tmp_uh),
        ptr<scalar_t>(drop_mask));

  }));

  return { tmp_wx };
}

}  // anonymous namespace

void ligru_init(py::module& m) {
  m.def("ligru_forward", &ligru_forward, "Li-GRU forward", py::call_guard<py::gil_scoped_release>());
}

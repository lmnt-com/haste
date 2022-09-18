// Copyright 2022 Adel Moumen, All Rights Reserved.
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

// Modified from https://github.com/lmnt-com/haste/blob/master/frameworks/pytorch/support.cc

#include <torch/extension.h>

void ligru_1_0_init(py::module &);
void ligru_2_0_init(py::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ligru_2_0_init(m);
  ligru_1_0_init(m);
}

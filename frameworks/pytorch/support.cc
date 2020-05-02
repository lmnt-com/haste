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

#include <torch/extension.h>

void gru_init(py::module&);
void indrnn_init(py::module&);
void lstm_init(py::module&);
void layer_norm_gru_init(py::module&);
void layer_norm_indrnn_init(py::module&);
void layer_norm_lstm_init(py::module&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gru_init(m);
  indrnn_init(m);
  lstm_init(m);
  layer_norm_gru_init(m);
  layer_norm_indrnn_init(m);
  layer_norm_lstm_init(m);
}

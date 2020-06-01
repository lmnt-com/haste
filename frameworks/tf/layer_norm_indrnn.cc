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

#include <cuda_runtime_api.h>

#include "arena.h"
#include "haste.h"
#include "support.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/stream.h"

using namespace tensorflow;

using tensorflow::se::Stream;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

namespace layer_norm = haste::v0::layer_norm;
namespace layer_norm_indrnn = haste::v0::layer_norm_indrnn;

// Define the interface and shape function for the op.
REGISTER_OP("HasteLayerNormIndrnn")
    .Attr("R: {float, double}")         // Some real number type.
    .Attr("training: bool")
    .Attr("zoneout_prob: float")
    .Input("x: R")                      // [T,N,C]
    .Input("kernel: R")                 // [C,H]
    .Input("recurrent_scale: R")        // [H]
    .Input("bias: R")                   // [H]
    .Input("gamma: R")                  // [2,H]
    .Input("zoneout_mask: R")           // [T,N,H]
    .Output("h: R")                     // [T+1,N,H]
    .Output("cache: R")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      ShapeHandle kernel_shape;
      ShapeHandle recurrent_shape;
      ShapeHandle bias_shape;
      ShapeHandle gamma_shape;
      ShapeHandle zoneout_mask_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &kernel_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &recurrent_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bias_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &gamma_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 3, &zoneout_mask_shape));

      const DimensionHandle time_steps = c->Dim(input_shape, 0);
      const DimensionHandle batch_size = c->Dim(input_shape, 1);
      const DimensionHandle hidden_size = c->Dim(recurrent_shape, 0);
      DimensionHandle time_steps_plus_1;

      TF_RETURN_IF_ERROR(c->Add(time_steps, 1, &time_steps_plus_1));

      c->set_output(0, c->MakeShape({ time_steps_plus_1, batch_size, hidden_size }));
      c->set_output(1, c->UnknownShapeOfRank(1));
      return Status::OK();
    });

template<typename T>
struct HasteLayerNormIndrnnOp : public OpKernel {
  explicit HasteLayerNormIndrnnOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
    OP_REQUIRES_OK(context, context->GetAttr("zoneout_prob", &zoneout_prob_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& kernel = context->input(1);
    const Tensor& recurrent_scale = context->input(2);
    const Tensor& bias = context->input(3);
    const Tensor& gamma = context->input(4);
    const Tensor& zoneout_mask = context->input(5);

    const auto time_steps = input.shape().dim_size(0);
    const auto batch_size = input.shape().dim_size(1);
    const auto input_size = input.shape().dim_size(2);
    const auto hidden_size = recurrent_scale.shape().dim_size(0);
    const bool has_zoneout = zoneout_prob_ && zoneout_mask.NumElements();
    const auto data_type = DataTypeToEnum<T>::value;

    const TensorShape output_shape = { time_steps + 1, batch_size, hidden_size };
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    Tensor workspace;
    const TensorShape workspace_shape = { time_steps, batch_size, hidden_size };
    OP_REQUIRES_OK(context, context->allocate_temp(data_type, workspace_shape, &workspace));

    const ArenaLayout<T> memory_layout = {
      { "act_Wx", { time_steps, batch_size, hidden_size } },
      { "act_Wx_norm_cache", { time_steps, batch_size, 2 } },
    };

    Tensor* output_cache = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, { memory_layout.NumElements() }, &output_cache));

    Arena<T> memory = memory_layout.Realize(output_cache->flat<T>().data());
    TensorView<T> act_Wx = memory["act_Wx"];
    TensorView<T> act_Wx_norm_cache = memory["act_Wx_norm_cache"];

    cudaMemset(output->flat<T>().data(), 0, output->AllocatedBytes());

    layer_norm::ForwardPass<T> layer_norm1(
        time_steps * batch_size,
        hidden_size,
        gamma.SubSlice(0).unaligned_flat<T>().data(),
        nullptr,
        act_Wx_norm_cache.data());

    layer_norm_indrnn::ForwardPass<T> forward(
        training_,
        batch_size,
        input_size,
        hidden_size,
        GetCublasHandle(context));

    forward.Run(
        time_steps,
        kernel.flat<T>().data(),
        recurrent_scale.flat<T>().data(),
        bias.flat<T>().data(),
        input.flat<T>().data(),
        output->flat<T>().data(),
        workspace.flat<T>().data(),
        act_Wx.data(),
        layer_norm1,
        has_zoneout ? zoneout_prob_ : 0.0f,
        has_zoneout ? zoneout_mask.flat<T>().data() : nullptr);
  }

  private:
    bool training_;
    float zoneout_prob_;
};

REGISTER_GPU_KERNEL(HasteLayerNormIndrnn, float);
REGISTER_GPU_KERNEL(HasteLayerNormIndrnn, double);

REGISTER_OP("HasteLayerNormIndrnnGrad")
    .Attr("R: {float, double}")
    .Input("x_t: R")                   // [C,N,T]
    .Input("kernel_t: R")              // [4,C]
    .Input("recurrent_scale: R")       // [H]
    .Input("bias: R")                  // [H]
    .Input("gamma: R")                 // [2,H]
    .Input("zoneout_mask: R")          // [T,N,H]
    .Input("h: R")                     // [T+1,N,H]
    .Input("cache: R")                 // [?]
    .Input("dh_new: R")                // [T+1,N,H]
    .Output("dx: R")                   // [T,N,C]
    .Output("dw: R")                   // [C,H]
    .Output("dr: R")                   // [H]
    .Output("db: R")                   // [H]
    .Output("dgamma: R")               // [H]
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x_shape;
      ShapeHandle kernel_shape;
      ShapeHandle recurrent_shape;
      ShapeHandle bias_shape;
      ShapeHandle gamma_shape;
      ShapeHandle zoneout_mask_shape;
      ShapeHandle h_shape;
      ShapeHandle cache_shape;
      ShapeHandle dh_new_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &kernel_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &recurrent_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bias_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 1, &gamma_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 3, &zoneout_mask_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 3, &h_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &cache_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 3, &dh_new_shape));

      DimensionHandle input_size = c->Dim(x_shape, 0);
      DimensionHandle time_steps = c->Dim(x_shape, 1);
      DimensionHandle batch_size = c->Dim(x_shape, 2);
      DimensionHandle hidden_size = c->Dim(recurrent_shape, 0);

      c->set_output(0, c->MakeShape({ time_steps, batch_size, input_size }));
      c->set_output(1, c->MakeShape({ input_size, hidden_size }));
      c->set_output(2, recurrent_shape);
      c->set_output(3, bias_shape);
      c->set_output(4, gamma_shape);
      return Status::OK();
    });

template<typename T>
struct HasteLayerNormIndrnnGradOp : public OpKernel {
  explicit HasteLayerNormIndrnnGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& kernel = context->input(1);
    const Tensor& recurrent_scale = context->input(2);
    const Tensor& bias = context->input(3);
    const Tensor& gamma = context->input(4);
    const Tensor& zoneout_mask = context->input(5);
    const Tensor& h_vector = context->input(6);
    const Tensor& cache_input = context->input(7);
    const Tensor& dh_new = context->input(8);

    const auto input_size = input.shape().dim_size(0);
    const auto time_steps = input.shape().dim_size(1);
    const auto batch_size = input.shape().dim_size(2);
    const auto hidden_size = recurrent_scale.shape().dim_size(0);
    const bool has_zoneout = !!zoneout_mask.NumElements();
    const auto data_type = DataTypeToEnum<T>::value;

    const ArenaLayout<T> memory_layout = {
      { "act_Wx", { time_steps, batch_size, hidden_size } },
      { "act_Wx_norm_cache", { time_steps, batch_size, 2 } },
    };

    assert(cache_input.shape().num_elements() == memory_layout.NumElements());

    Arena<T> memory = memory_layout.Realize(const_cast<T*>(cache_input.flat<T>().data()));
    TensorView<T> act_Wx = memory["act_Wx"];
    TensorView<T> act_Wx_norm_cache = memory["act_Wx_norm_cache"];

    // Can be uninitialized. Output only, no accumulation.
    const TensorShape dx_shape = { time_steps, batch_size, input_size };
    Tensor* dx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, dx_shape, &dx));

    // Needs to be initialized to 0.
    const TensorShape dW_shape = { input_size, hidden_size };
    Tensor* dW = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, dW_shape, &dW));

    // Needs to be initialized to 0.
    const TensorShape du_shape = { hidden_size };
    Tensor* du = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, du_shape, &du));

    // Needs to be initialized to 0.
    Tensor* db = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, bias.shape(), &db));

    // Needs to be initialized to 0.
    Tensor* dgamma = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(4, gamma.shape(), &dgamma));

    // Needs to be initialized to 0.
    const TensorShape dh_shape = { batch_size, hidden_size };
    Tensor dh;
    OP_REQUIRES_OK(context, context->allocate_temp(data_type, dh_shape, &dh));

    const TensorShape workspace_shape = { time_steps, batch_size, hidden_size };
    Tensor workspace;
    OP_REQUIRES_OK(context, context->allocate_temp(data_type, workspace_shape, &workspace));

    cudaMemset(dW->flat<T>().data(), 0, dW->AllocatedBytes());
    cudaMemset(du->flat<T>().data(), 0, du->AllocatedBytes());
    cudaMemset(db->flat<T>().data(), 0, db->AllocatedBytes());
    cudaMemset(dgamma->flat<T>().data(), 0, dgamma->AllocatedBytes());
    cudaMemset(dh.flat<T>().data(), 0, dh.AllocatedBytes());

    layer_norm::BackwardPass<T> layer_norm1(
        time_steps * batch_size,
        hidden_size,
        gamma.SubSlice(0).unaligned_flat<T>().data(),
        nullptr,
        act_Wx.data(),
        dgamma->SubSlice(0).unaligned_flat<T>().data(),
        nullptr,
        act_Wx_norm_cache.data());

    layer_norm_indrnn::BackwardPass<T> backward(
        batch_size,
        input_size,
        hidden_size,
        GetCublasHandle(context));

    backward.Run(
        time_steps,
        kernel.flat<T>().data(),
        recurrent_scale.flat<T>().data(),
        bias.flat<T>().data(),
        input.flat<T>().data(),
        h_vector.flat<T>().data(),
        dh_new.flat<T>().data(),
        dx->flat<T>().data(),
        dW->flat<T>().data(),
        du->flat<T>().data(),
        db->flat<T>().data(),
        dh.flat<T>().data(),
        workspace.flat<T>().data(),
        layer_norm1,
        has_zoneout ? zoneout_mask.flat<T>().data() : nullptr);
  }
};

REGISTER_GPU_KERNEL(HasteLayerNormIndrnnGrad, float);
REGISTER_GPU_KERNEL(HasteLayerNormIndrnnGrad, double);

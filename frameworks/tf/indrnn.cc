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

#include "haste.h"
#include "support.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/stream.h"

using namespace tensorflow;

using haste::v0::indrnn::ForwardPass;
using haste::v0::indrnn::BackwardPass;
using tensorflow::se::Stream;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

// Define the interface and shape function for the op.
REGISTER_OP("HasteIndrnn")
    .Attr("R: {float, double}")         // Some real number type.
    .Attr("training: bool")
    .Attr("zoneout_prob: float")
    .Input("x: R")                      // [T,N,C]
    .Input("kernel: R")                 // [C,H]
    .Input("recurrent_scale: R")        // [H]
    .Input("bias: R")                   // [H]
    .Input("zoneout_mask: R")           // [T,N,H]
    .Output("h: R")                     // [T+1,N,H]
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      ShapeHandle kernel_shape;
      ShapeHandle recurrent_shape;
      ShapeHandle bias_shape;
      ShapeHandle zoneout_mask_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &kernel_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &recurrent_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bias_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 3, &zoneout_mask_shape));

      const DimensionHandle time_steps = c->Dim(input_shape, 0);
      const DimensionHandle batch_size = c->Dim(input_shape, 1);
      const DimensionHandle hidden_size = c->Dim(recurrent_shape, 0);
      DimensionHandle time_steps_plus_1;

      TF_RETURN_IF_ERROR(c->Add(time_steps, 1, &time_steps_plus_1));

      c->set_output(0, c->MakeShape({ time_steps_plus_1, batch_size, hidden_size }));
      return Status::OK();
    });

template<typename T>
struct HasteIndrnnOp : public OpKernel {
  explicit HasteIndrnnOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
    OP_REQUIRES_OK(context, context->GetAttr("zoneout_prob", &zoneout_prob_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& kernel = context->input(1);
    const Tensor& recurrent_scale = context->input(2);
    const Tensor& bias = context->input(3);
    const Tensor& zoneout_mask = context->input(4);

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

    cudaMemset(output->flat<T>().data(), 0, output->AllocatedBytes());

    ForwardPass<T> forward(
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
        has_zoneout ? zoneout_prob_ : 0.0f,
        has_zoneout ? zoneout_mask.flat<T>().data() : nullptr);
  }

  private:
    bool training_;
    float zoneout_prob_;
};

REGISTER_GPU_KERNEL(HasteIndrnn, float);
REGISTER_GPU_KERNEL(HasteIndrnn, double);

REGISTER_OP("HasteIndrnnGrad")
    .Attr("R: {float, double}")
    .Input("x_t: R")                   // [C,N,T]
    .Input("kernel_t: R")              // [4,C]
    .Input("recurrent_scale: R")       // [H]
    .Input("bias: R")                  // [H]
    .Input("zoneout_mask: R")          // [T,N,H]
    .Input("h: R")                     // [T+1,N,H]
    .Input("dh_new: R")                // [T+1,N,H]
    .Output("dx: R")                   // [T,N,C]
    .Output("dw: R")                   // [C,H]
    .Output("dr: R")                   // [H]
    .Output("db: R")                   // [H]
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x_shape;
      ShapeHandle kernel_shape;
      ShapeHandle recurrent_shape;
      ShapeHandle bias_shape;
      ShapeHandle zoneout_mask_shape;
      ShapeHandle h_shape;
      ShapeHandle dh_new_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &kernel_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &recurrent_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bias_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 3, &zoneout_mask_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 3, &h_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 3, &dh_new_shape));

      DimensionHandle input_size = c->Dim(x_shape, 0);
      DimensionHandle time_steps = c->Dim(x_shape, 1);
      DimensionHandle batch_size = c->Dim(x_shape, 2);
      DimensionHandle hidden_size = c->Dim(recurrent_shape, 0);

      c->set_output(0, c->MakeShape({ time_steps, batch_size, input_size }));
      c->set_output(1, c->MakeShape({ input_size, hidden_size }));
      c->set_output(2, recurrent_shape);
      c->set_output(3, bias_shape);
      return Status::OK();
    });

template<typename T>
struct HasteIndrnnGradOp : public OpKernel {
  explicit HasteIndrnnGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& kernel = context->input(1);
    const Tensor& recurrent_scale = context->input(2);
    const Tensor& bias = context->input(3);
    const Tensor& zoneout_mask = context->input(4);
    const Tensor& h_vector = context->input(5);
    const Tensor& dh_new = context->input(6);

    const auto input_size = input.shape().dim_size(0);
    const auto time_steps = input.shape().dim_size(1);
    const auto batch_size = input.shape().dim_size(2);
    const auto hidden_size = recurrent_scale.shape().dim_size(0);
    const bool has_zoneout = !!zoneout_mask.NumElements();
    const auto data_type = DataTypeToEnum<T>::value;

    // Can be uninitialized. Output only, no accumulation.
    const TensorShape dx_shape = { time_steps, batch_size, input_size };
    Tensor* dx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, dx_shape, &dx));

    // Needs to be initialized to 0.
    const TensorShape dW_shape = { input_size, hidden_size };
    Tensor* dW = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, dW_shape, &dW));

    // Needs to be initialized to 0.
    const TensorShape du_shape = { hidden_size, hidden_size };
    Tensor* du = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, du_shape, &du));

    // Needs to be initialized to 0.
    const TensorShape db_shape = { hidden_size };
    Tensor* db = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, db_shape, &db));

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
    cudaMemset(dh.flat<T>().data(), 0, dh.AllocatedBytes());

    BackwardPass<T> backward = BackwardPass<T>(
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
        has_zoneout ? zoneout_mask.flat<T>().data() : nullptr);
  }
};

REGISTER_GPU_KERNEL(HasteIndrnnGrad, float);
REGISTER_GPU_KERNEL(HasteIndrnnGrad, double);

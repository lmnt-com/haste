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

using haste::v0::lstm::ForwardPass;
using haste::v0::lstm::BackwardPass;
using tensorflow::se::Stream;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

// Define the interface and shape function for the op.
REGISTER_OP("HasteLstm")
    .Attr("R: {float, double}")         // Some real number type.
    .Attr("training: bool")
    .Attr("zoneout_prob: float")
    .Input("x: R")                      // [T,N,C]
    .Input("kernel: R")                 // [C,H*4]
    .Input("recurrent_kernel: R")       // [H,H*4]
    .Input("bias: R")                   // [H*4]
    .Input("zoneout_mask: R")           // [T,N,H]
    .Output("h: R")                     // [T,N,H]
    .Output("c: R")                     // [T,N,H]
    .Output("v: R")                     // [T,N,H*4]
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      ShapeHandle kernel_shape;
      ShapeHandle recurrent_shape;
      ShapeHandle bias_shape;
      ShapeHandle zoneout_mask_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &kernel_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &recurrent_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bias_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 3, &zoneout_mask_shape));

      const DimensionHandle time_steps = c->Dim(input_shape, 0);
      const DimensionHandle batch_size = c->Dim(input_shape, 1);
      const DimensionHandle hidden_size = c->Dim(recurrent_shape, 0);
      DimensionHandle time_steps_plus_1;
      DimensionHandle hidden_size_4;

      TF_RETURN_IF_ERROR(c->Add(time_steps, 1, &time_steps_plus_1));
      TF_RETURN_IF_ERROR(c->Multiply(hidden_size, 4, &hidden_size_4));

      c->set_output(0, c->MakeShape({ time_steps_plus_1, batch_size, hidden_size }));
      c->set_output(1, c->MakeShape({ time_steps_plus_1, batch_size, hidden_size }));
      c->set_output(2, c->MakeShape({ time_steps, batch_size, hidden_size_4 }));
      return Status::OK();
    });

template<typename T>
struct HasteLstmOp : public OpKernel {
  explicit HasteLstmOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
    OP_REQUIRES_OK(context, context->GetAttr("zoneout_prob", &zoneout_prob_));
  }

  // When running on GPU, TF backs all inputs and outputs with device memory
  // and not host memory. We don't need to do explicit memory copies or allocations
  // for the inputs and outputs.
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& kernel = context->input(1);
    const Tensor& recurrent_kernel = context->input(2);
    const Tensor& bias = context->input(3);
    const Tensor& zoneout_mask = context->input(4);

    const auto time_steps = input.shape().dim_size(0);
    const auto batch_size = input.shape().dim_size(1);
    const auto input_size = input.shape().dim_size(2);
    const auto hidden_size = recurrent_kernel.shape().dim_size(0);
    const bool has_zoneout = zoneout_prob_ && zoneout_mask.NumElements();
    const auto data_type = DataTypeToEnum<T>::value;

    OP_REQUIRES(context, input_size == kernel.shape().dim_size(0),
        errors::InvalidArgument("input[2] and kernel[0] dimensions must match. Found ",
            input_size, " and ", kernel.shape().dim_size(0)));

    const TensorShape output_shape = { time_steps + 1, batch_size, hidden_size };
    const TensorShape activations_shape = { time_steps, batch_size, hidden_size * 4 };

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    Tensor* output_cell_state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &output_cell_state));

    Tensor output_v_temp;
    Tensor* output_v = nullptr;
    if (training_) {
      OP_REQUIRES_OK(context, context->allocate_output(2, activations_shape, &output_v));
    } else {
      // Return an empty tensor in inference mode and provide temp memory
      // to the forward pass instead.
      OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({ 0 }), &output_v));
      OP_REQUIRES_OK(context, context->allocate_temp(data_type, activations_shape, &output_v_temp));
      output_v = &output_v_temp;
    }

    Tensor tmp_Rh;
    const TensorShape tmp_Rh_shape = { batch_size, 4 * hidden_size };
    OP_REQUIRES_OK(context, context->allocate_temp(data_type, tmp_Rh_shape, &tmp_Rh));
    cudaMemset(output->flat<T>().data(), 0, output->AllocatedBytes());
    cudaMemset(output_cell_state->flat<T>().data(), 0, output_cell_state->AllocatedBytes());

    ForwardPass<T> forward = ForwardPass<T>(
        training_,
        batch_size,
        input_size,
        hidden_size,
        GetCublasHandle(context));

    forward.Run(
        time_steps,
        kernel.flat<T>().data(),
        recurrent_kernel.flat<T>().data(),
        bias.flat<T>().data(),
        input.flat<T>().data(),
        output->flat<T>().data(),
        output_cell_state->flat<T>().data(),
        output_v->flat<T>().data(),
        tmp_Rh.flat<T>().data(),
        has_zoneout ? zoneout_prob_ : 0.0f,
        has_zoneout ? zoneout_mask.flat<T>().data() : nullptr);
  }

  private:
    bool training_;
    float zoneout_prob_;
};

REGISTER_GPU_KERNEL(HasteLstm, float);
REGISTER_GPU_KERNEL(HasteLstm, double);

REGISTER_OP("HasteLstmGrad")
    .Attr("R: {float, double}")
    .Input("x_t: R")                   // [C,N,T]
    .Input("kernel_t: R")              // [H*4,C]
    .Input("recurrent_kernel_t: R")    // [H*4,H]
    .Input("bias: R")                  // [H*4]
    .Input("h: R")                     // [T,N,H]
    .Input("c: R")                     // [T,N,H]
    .Input("v: R")                     // [T,N,H*4]
    .Input("dh_new: R")                // [T,N,H]
    .Input("dc_new: R")                // [T,N,H]
    .Input("zoneout_mask: R")          // [T,N,H]
    .Output("dx: R")                   // [T,N,C]
    .Output("dw: R")                   // [C,H*4]
    .Output("dr: R")                   // [H,H*4]
    .Output("db: R")                   // [H*4]
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x_shape;
      ShapeHandle kernel_shape;
      ShapeHandle recurrent_kernel_shape;
      ShapeHandle bias_shape;
      ShapeHandle h_shape;
      ShapeHandle c_shape;
      ShapeHandle v_shape;
      ShapeHandle dh_new_shape;
      ShapeHandle dc_new_shape;
      ShapeHandle zoneout_mask_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &x_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &kernel_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &recurrent_kernel_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bias_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 3, &h_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 3, &c_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 3, &v_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 3, &dh_new_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 3, &dc_new_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 3, &zoneout_mask_shape));

      DimensionHandle input_size = c->Dim(x_shape, 0);
      DimensionHandle time_steps = c->Dim(x_shape, 1);
      DimensionHandle batch_size = c->Dim(x_shape, 2);
      DimensionHandle hidden_size = c->Dim(recurrent_kernel_shape, 1);
      DimensionHandle hidden_size_4;

      TF_RETURN_IF_ERROR(c->Multiply(hidden_size, 4, &hidden_size_4));

      c->set_output(0, c->MakeShape({ time_steps, batch_size, input_size }));
      c->set_output(1, c->MakeShape({ input_size, hidden_size_4 }));
      c->set_output(2, c->MakeShape({ hidden_size, hidden_size_4 }));
      c->set_output(3, bias_shape);
      return Status::OK();
    });

template<typename T>
struct HasteLstmGradOp : public OpKernel {
  explicit HasteLstmGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& kernel = context->input(1);
    const Tensor& recurrent_kernel = context->input(2);
    const Tensor& bias = context->input(3);
    const Tensor& h_vector = context->input(4);
    const Tensor& c_vector = context->input(5);
    const Tensor& dv = context->input(6);
    const Tensor& dh_new = context->input(7);
    const Tensor& dc_new = context->input(8);
    const Tensor& zoneout_mask = context->input(9);

    const auto input_size = input.shape().dim_size(0);
    const auto time_steps = input.shape().dim_size(1);
    const auto batch_size = input.shape().dim_size(2);
    const auto hidden_size = recurrent_kernel.shape().dim_size(1);
    const bool has_zoneout = !!zoneout_mask.NumElements();
    const auto data_type = DataTypeToEnum<T>::value;

    // Can be uninitialized. Output only, no accumulation.
    const TensorShape dx_shape = { time_steps, batch_size, input_size };
    Tensor* dx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, dx_shape, &dx));

    // Needs to be initialized to 0.
    const TensorShape dW_shape = { input_size, hidden_size * 4 };
    Tensor* dW = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, dW_shape, &dW));

    // Needs to be initialized to 0.
    const TensorShape dR_shape = { hidden_size, hidden_size * 4 };
    Tensor* dR = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, dR_shape, &dR));

    // Needs to be initialized to 0.
    const TensorShape db_shape = { hidden_size * 4 };
    Tensor* db = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, db_shape, &db));

    // Needs to be initialized to 0.
    const TensorShape dh_shape = { batch_size, hidden_size };
    Tensor dh;
    OP_REQUIRES_OK(context, context->allocate_temp(data_type, dh_shape, &dh));

    // Needs to be initialized to 0.
    const TensorShape dc_shape = { batch_size, hidden_size };
    Tensor dc;
    OP_REQUIRES_OK(context, context->allocate_temp(data_type, dc_shape, &dc));

    cudaMemset(dW->flat<T>().data(), 0, dW->AllocatedBytes());
    cudaMemset(dR->flat<T>().data(), 0, dR->AllocatedBytes());
    cudaMemset(db->flat<T>().data(), 0, db->AllocatedBytes());
    cudaMemset(dh.flat<T>().data(), 0, dh.AllocatedBytes());
    cudaMemset(dc.flat<T>().data(), 0, dc.AllocatedBytes());

    BackwardPass<T> backward = BackwardPass<T>(
        batch_size,
        input_size,
        hidden_size,
        GetCublasHandle(context));

    backward.Run(
        time_steps,
        kernel.flat<T>().data(),
        recurrent_kernel.flat<T>().data(),
        bias.flat<T>().data(),
        input.flat<T>().data(),
        h_vector.flat<T>().data(),
        c_vector.flat<T>().data(),
        dh_new.flat<T>().data(),
        dc_new.flat<T>().data(),
        dx->flat<T>().data(),
        dW->flat<T>().data(),
        dR->flat<T>().data(),
        db->flat<T>().data(),
        dh.flat<T>().data(),
        dc.flat<T>().data(),
        const_cast<T*>(dv.flat<T>().data()),
        has_zoneout ? zoneout_mask.flat<T>().data() : nullptr);
  }
};

REGISTER_GPU_KERNEL(HasteLstmGrad, float);
REGISTER_GPU_KERNEL(HasteLstmGrad, double);

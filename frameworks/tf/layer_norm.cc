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

#include "haste.h"
#include "support.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/stream_executor/stream.h"

using namespace tensorflow;

using haste::v0::layer_norm::ForwardPass;
using haste::v0::layer_norm::BackwardPass;
using tensorflow::se::Stream;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("HasteLayerNorm")
    .Attr("R: {float, double}")
    .Input("x: R")
    .Input("alpha: R")
    .Input("beta: R")
    .Output("y: R")
    .Output("cache: R")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_shape;
      ShapeHandle alpha_shape;
      ShapeHandle beta_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &alpha_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &beta_shape));

      c->set_output(0, input_shape);
      c->set_output(1, c->UnknownShapeOfRank(2));
      return Status::OK();
    });

template<typename T>
struct HasteLayerNormOp : public OpKernel {
  explicit HasteLayerNormOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& alpha = context->input(1);
    const Tensor& beta = context->input(2);

    const auto batch_size = x.shape().dim_size(0);
    const auto hidden_size = x.shape().dim_size(1);

    Tensor* y = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &y));

    Tensor* cache = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, { batch_size, 2 }, &cache));

    ForwardPass<T> forward(
        batch_size,
        hidden_size,
        alpha.flat<T>().data(),
        beta.flat<T>().data(),
        cache->flat<T>().data());

    forward.Run(GetCudaStream(context), x.flat<T>().data(), y->flat<T>().data());
  }
};

REGISTER_GPU_KERNEL(HasteLayerNorm, float);
REGISTER_GPU_KERNEL(HasteLayerNorm, double);

REGISTER_OP("HasteLayerNormGrad")
    .Attr("R: {float, double}")
    .Input("x: R")
    .Input("alpha: R")
    .Input("beta: R")
    .Input("dy: R")
    .Input("cache: R")
    .Output("dx: R")
    .Output("dalpha: R")
    .Output("dbeta: R")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x_shape;
      ShapeHandle alpha_shape;
      ShapeHandle beta_shape;
      ShapeHandle dy_shape;
      ShapeHandle cache_shape;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &x_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &alpha_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &beta_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &dy_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 2, &cache_shape));

      c->set_output(0, x_shape);
      c->set_output(1, alpha_shape);
      c->set_output(2, beta_shape);
      return Status::OK();
    });

template<typename T>
struct HasteLayerNormGradOp : public OpKernel {
  explicit HasteLayerNormGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& alpha = context->input(1);
    const Tensor& beta = context->input(2);
    const Tensor& dy = context->input(3);
    const Tensor& cache = context->input(4);

    const auto batch_size = x.shape().dim_size(0);
    const auto hidden_size = x.shape().dim_size(1);
    const auto cache_shape = context->input(4).shape();
    const auto data_type = DataTypeToEnum<T>::value;

    Tensor* dx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &dx));

    Tensor* dalpha = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, alpha.shape(), &dalpha));

    Tensor* dbeta = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, beta.shape(), &dbeta));

    cudaMemset(dalpha->flat<T>().data(), 0, dalpha->AllocatedBytes());
    cudaMemset(dbeta->flat<T>().data(), 0, dbeta->AllocatedBytes());

    BackwardPass<T> backward(
        batch_size,
        hidden_size,
        alpha.flat<T>().data(),
        beta.flat<T>().data(),
        x.flat<T>().data(),
        dalpha->flat<T>().data(),
        dbeta->flat<T>().data(),
        const_cast<T*>(cache.flat<T>().data()));

    backward.Run(GetCudaStream(context), dy.flat<T>().data(), dx->flat<T>().data());
  }
};

REGISTER_GPU_KERNEL(HasteLayerNormGrad, float);
REGISTER_GPU_KERNEL(HasteLayerNormGrad, double);

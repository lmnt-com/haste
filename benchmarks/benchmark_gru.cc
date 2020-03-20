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

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <getopt.h>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "../examples/device_ptr.h"
#include "cudnn_wrappers.h"
#include "haste.h"

using haste::v0::gru::BackwardPass;
using haste::v0::gru::ForwardPass;
using std::string;

using Tensor1 = Eigen::Tensor<float, 1>;
using Tensor2 = Eigen::Tensor<float, 2>;
using Tensor3 = Eigen::Tensor<float, 3>;

static constexpr int DEFAULT_SAMPLE_SIZE = 10;
static constexpr int DEFAULT_TIME_STEPS = 50;

static cudnnHandle_t g_cudnn_handle;
static cublasHandle_t g_blas_handle;

float TimeLoop(std::function<void()> fn, int iterations) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iterations; ++i)
    fn();
  float elapsed_ms;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return elapsed_ms / iterations;
}

float CudnnInference(
    int sample_size,
    const Tensor2& W,
    const Tensor2& R,
    const Tensor1& bx,
    const Tensor1& br,
    const Tensor3& x) {
  const int time_steps = x.dimension(2);
  const int batch_size = x.dimension(1);
  const int input_size = x.dimension(0);
  const int hidden_size = R.dimension(1);

  device_ptr<Tensor3> x_dev(x);

  device_ptr<Tensor2> h_dev(batch_size * hidden_size);
  device_ptr<Tensor2> c_dev(batch_size * hidden_size);
  device_ptr<Tensor3> y_dev(time_steps * batch_size * hidden_size);
  device_ptr<Tensor2> h_out_dev(batch_size * hidden_size);
  device_ptr<Tensor2> c_out_dev(batch_size * hidden_size);

  h_dev.zero();
  c_dev.zero();

  // Descriptors all the way down. Nice.
  RnnDescriptor<float> rnn_descriptor(g_cudnn_handle, hidden_size, CUDNN_GRU);

  TensorDescriptorArray<float> x_descriptors(time_steps, { batch_size, input_size, 1 });
  TensorDescriptorArray<float> y_descriptors(time_steps, { batch_size, hidden_size, 1 });

  auto h_descriptor = TensorDescriptor<float>({ 1, batch_size, hidden_size });
  auto c_descriptor = TensorDescriptor<float>({ 1, batch_size, hidden_size });
  auto h_out_descriptor = TensorDescriptor<float>({ 1, batch_size, hidden_size });
  auto c_out_descriptor = TensorDescriptor<float>({ 1, batch_size, hidden_size });

  size_t workspace_size;
  cudnnGetRNNWorkspaceSize(
      g_cudnn_handle,
      *rnn_descriptor,
      time_steps,
      &x_descriptors,
      &workspace_size);
  auto workspace_dev = device_ptr<Tensor1>::NewByteSized(workspace_size);

  size_t w_count;
  cudnnGetRNNParamsSize(
      g_cudnn_handle,
      *rnn_descriptor,
      *&x_descriptors,
      &w_count,
      CUDNN_DATA_FLOAT);

  auto w_dev = device_ptr<Tensor1>::NewByteSized(w_count);
  FilterDescriptor<float> w_descriptor(w_dev.Size());

  float ms = TimeLoop([&]() {
    cudnnRNNForwardInference(
        g_cudnn_handle,
        *rnn_descriptor,
        time_steps,
        &x_descriptors,
        x_dev.data,
        *h_descriptor,
        h_dev.data,
        *c_descriptor,
        c_dev.data,
        *w_descriptor,
        w_dev.data,
        &y_descriptors,
        y_dev.data,
        *h_out_descriptor,
        h_out_dev.data,
        *c_out_descriptor,
        c_out_dev.data,
        workspace_dev.data,
        workspace_size);
  }, sample_size);
  return ms;
}

float CudnnTrain(
    int sample_size,
    const Tensor2& W,
    const Tensor2& R,
    const Tensor1& bx,
    const Tensor1& br,
    const Tensor3& x,
    const Tensor3& dh) {
  const int time_steps = x.dimension(2);
  const int batch_size = x.dimension(1);
  const int input_size = x.dimension(0);
  const int hidden_size = R.dimension(1);

  device_ptr<Tensor3> y_dev(time_steps * batch_size * hidden_size);
  device_ptr<Tensor3> dy_dev(time_steps * batch_size * hidden_size);
  device_ptr<Tensor2> dhy_dev(batch_size * hidden_size);
  device_ptr<Tensor2> dcy_dev(batch_size * hidden_size);
  device_ptr<Tensor2> hx_dev(batch_size * hidden_size);
  device_ptr<Tensor2> cx_dev(batch_size * hidden_size);
  device_ptr<Tensor2> dx_dev(time_steps * batch_size * input_size);
  device_ptr<Tensor2> dhx_dev(batch_size * hidden_size);
  device_ptr<Tensor2> dcx_dev(batch_size * hidden_size);

  RnnDescriptor<float> rnn_descriptor(g_cudnn_handle, hidden_size, CUDNN_GRU);
  TensorDescriptorArray<float> y_descriptors(time_steps, { batch_size, hidden_size, 1 });
  TensorDescriptorArray<float> dy_descriptors(time_steps, { batch_size, hidden_size, 1 });
  TensorDescriptorArray<float> dx_descriptors(time_steps, { batch_size, input_size, 1 });

  TensorDescriptor<float> dhy_descriptor({ 1, batch_size, hidden_size });
  TensorDescriptor<float> dcy_descriptor({ 1, batch_size, hidden_size });
  TensorDescriptor<float> hx_descriptor({ 1, batch_size, hidden_size });
  TensorDescriptor<float> cx_descriptor({ 1, batch_size, hidden_size });
  TensorDescriptor<float> dhx_descriptor({ 1, batch_size, hidden_size });
  TensorDescriptor<float> dcx_descriptor({ 1, batch_size, hidden_size });

  size_t workspace_size = 0;
  cudnnGetRNNWorkspaceSize(
      g_cudnn_handle,
      *rnn_descriptor,
      time_steps,
      &dx_descriptors,
      &workspace_size);
  auto workspace_dev = device_ptr<Tensor1>::NewByteSized(workspace_size);

  size_t w_count = 0;
  cudnnGetRNNParamsSize(
      g_cudnn_handle,
      *rnn_descriptor,
      *&dx_descriptors,
      &w_count,
      CUDNN_DATA_FLOAT);

  auto w_dev = device_ptr<Tensor1>::NewByteSized(w_count);
  FilterDescriptor<float> w_descriptor(w_dev.Size());

  size_t reserve_size = 0;
  cudnnGetRNNTrainingReserveSize(
      g_cudnn_handle,
      *rnn_descriptor,
      time_steps,
      &dx_descriptors,
      &reserve_size);
  auto reserve_dev = device_ptr<Tensor1>::NewByteSized(reserve_size);

  float ms = TimeLoop([&]() {
    cudnnRNNForwardTraining(
        g_cudnn_handle,
        *rnn_descriptor,
        time_steps,
        &dx_descriptors,
        dx_dev.data,
        *hx_descriptor,
        hx_dev.data,
        *cx_descriptor,
        cx_dev.data,
        *w_descriptor,
        w_dev.data,
        &y_descriptors,
        y_dev.data,
        *dhy_descriptor,
        dhy_dev.data,
        *dcy_descriptor,
        dcy_dev.data,
        workspace_dev.data,
        workspace_size,
        reserve_dev.data,
        reserve_size);

    cudnnRNNBackwardData(
        g_cudnn_handle,
        *rnn_descriptor,
        time_steps,
        &y_descriptors,
        y_dev.data,
        &dy_descriptors,
        dy_dev.data,
        *dhy_descriptor,
        dhy_dev.data,
        *dcy_descriptor,
        dcy_dev.data,
        *w_descriptor,
        w_dev.data,
        *hx_descriptor,
        hx_dev.data,
        *cx_descriptor,
        cx_dev.data,
        &dx_descriptors,
        dx_dev.data,
        *dhx_descriptor,
        dhx_dev.data,
        *dcx_descriptor,
        dcx_dev.data,
        workspace_dev.data,
        workspace_size,
        reserve_dev.data,
        reserve_size);

    cudnnRNNBackwardWeights(
        g_cudnn_handle,
        *rnn_descriptor,
        time_steps,
        &dx_descriptors,
        dx_dev.data,
        *hx_descriptor,
        hx_dev.data,
        &y_descriptors,
        y_dev.data,
        workspace_dev.data,
        workspace_size,
        *w_descriptor,
        w_dev.data,
        reserve_dev.data,
        reserve_size);
  }, sample_size);
  return ms;
}

float HasteInference(
    int sample_size,
    const Tensor2& W,
    const Tensor2& R,
    const Tensor1& bx,
    const Tensor1& br,
    const Tensor3& x) {
  const int time_steps = x.dimension(2);
  const int batch_size = x.dimension(1);
  const int input_size = x.dimension(0);
  const int hidden_size = R.dimension(1);

  // Copy weights over to GPU.
  device_ptr<Tensor2> W_dev(W);
  device_ptr<Tensor2> R_dev(R);
  device_ptr<Tensor1> bx_dev(bx);
  device_ptr<Tensor1> br_dev(br);
  device_ptr<Tensor3> x_dev(x);

  device_ptr<Tensor3> h_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor2> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor2> tmp_Rh_dev(batch_size * hidden_size * 3);

  h_dev.zero();

  // Settle down the GPU and off we go!
  cudaDeviceSynchronize();
  float ms = TimeLoop([&]() {
    ForwardPass<float> forward(
        false,
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    forward.Run(
        time_steps,
        W_dev.data,
        R_dev.data,
        bx_dev.data,
        br_dev.data,
        x_dev.data,
        h_dev.data,
        nullptr,
        tmp_Wx_dev.data,
        tmp_Rh_dev.data,
        0.0f,
        nullptr);
  }, sample_size);
  return ms;
}

float HasteTrain(
    int sample_size,
    const Tensor2& W,
    const Tensor2& R,
    const Tensor1& bx,
    const Tensor1& br,
    const Tensor3& x,
    const Tensor3& dh) {
  const int time_steps = x.dimension(2);
  const int batch_size = x.dimension(1);
  const int input_size = x.dimension(0);
  const int hidden_size = R.dimension(1);

  device_ptr<Tensor2> W_dev(W);
  device_ptr<Tensor2> R_dev(R);
  device_ptr<Tensor3> x_dev(x);
  device_ptr<Tensor3> h_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor3> v_dev(time_steps * batch_size * hidden_size * 4);
  device_ptr<Tensor2> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor2> tmp_Rh_dev(batch_size * hidden_size * 3);

  device_ptr<Tensor2> W_t_dev(W);
  device_ptr<Tensor2> R_t_dev(R);
  device_ptr<Tensor1> bx_dev(bx);
  device_ptr<Tensor1> br_dev(br);
  device_ptr<Tensor3> x_t_dev(x);

  // These gradients should actually come "from above" but we're just allocating
  // a bunch of uninitialized memory and passing it in.
  device_ptr<Tensor3> dh_new_dev(dh);

  device_ptr<Tensor3> dx_dev(time_steps * batch_size * input_size);
  device_ptr<Tensor2> dW_dev(input_size * hidden_size * 3);
  device_ptr<Tensor2> dR_dev(hidden_size * hidden_size * 3);
  device_ptr<Tensor2> dbx_dev(hidden_size * 3);
  device_ptr<Tensor2> dbr_dev(hidden_size * 3);
  device_ptr<Tensor2> dh_dev(batch_size * hidden_size);
  device_ptr<Tensor3> dp_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor3> dq_dev(time_steps * batch_size * hidden_size * 3);

  ForwardPass<float> forward(
      true,
      batch_size,
      input_size,
      hidden_size,
      g_blas_handle);

  BackwardPass<float> backward(
      batch_size,
      input_size,
      hidden_size,
      g_blas_handle);

  static const float alpha = 1.0f;
  static const float beta = 0.0f;

  cudaDeviceSynchronize();
  float ms = TimeLoop([&]() {
    forward.Run(
        time_steps,
        W_dev.data,
        R_dev.data,
        bx_dev.data,
        br_dev.data,
        x_dev.data,
        h_dev.data,
        v_dev.data,
        tmp_Wx_dev.data,
        tmp_Rh_dev.data,
        0.0f,
        nullptr);

    // Haste needs `x`, `W`, and `R` to be transposed between the forward
    // pass and backward pass. Add these transposes in here to get a fair
    // measurement of the overall time it takes to run an entire training
    // loop.
    cublasSgeam(
        g_blas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        batch_size * time_steps, input_size,
        &alpha,
        x_dev.data, input_size,
        &beta,
        x_dev.data, batch_size * time_steps,
        x_t_dev.data, batch_size * time_steps);

    cublasSgeam(
        g_blas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        input_size, hidden_size * 3,
        &alpha,
        W_dev.data, hidden_size * 3,
        &beta,
        W_dev.data, input_size,
        W_t_dev.data, input_size);

    cublasSgeam(
        g_blas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_size, hidden_size * 3,
        &alpha,
        R_dev.data, hidden_size * 3,
        &beta,
        R_dev.data, hidden_size,
        R_t_dev.data, hidden_size);

    backward.Run(
        time_steps,
        W_t_dev.data,
        R_t_dev.data,
        bx_dev.data,
        br_dev.data,
        x_t_dev.data,
        h_dev.data,
        v_dev.data,
        dh_new_dev.data,
        dx_dev.data,
        dW_dev.data,
        dR_dev.data,
        dbx_dev.data,
        dbr_dev.data,
        dh_dev.data,
        dp_dev.data,
        dq_dev.data,
        nullptr);
  }, sample_size);
  return ms;
}

void usage(const char* name) {
  printf("Usage: %s [OPTION]...\n", name);
  printf("  -h, --help\n");
  printf("  -i, --implementation IMPL <haste|cudnn> (default: haste)\n");
  printf("  -m, --mode MODE           <inference|training> (default: training)\n");
  printf("  -s, --sample_size NUM     number of runs to average over (default: %d)\n",
      DEFAULT_SAMPLE_SIZE);
  printf("  -t, --time_steps NUM      number of time steps in RNN (default: %d)\n",
      DEFAULT_TIME_STEPS);
}

int main(int argc, char* const* argv) {
  srand(time(0));

  cudnnCreate(&g_cudnn_handle);
  cublasCreate(&g_blas_handle);

  static struct option long_options[] = {
    { "help", no_argument, 0, 'h' },
    { "implementation", required_argument, 0, 'i' },
    { "mode", required_argument, 0, 'm' },
    { "sample_size", required_argument, 0, 's' },
    { "time_steps", required_argument, 0, 't' },
    { 0, 0, 0, 0 }
  };

  int c;
  int opt_index;
  bool inference_flag = false;
  bool haste_flag = true;
  int sample_size = DEFAULT_SAMPLE_SIZE;
  int time_steps = DEFAULT_TIME_STEPS;
  while ((c = getopt_long(argc, argv, "hi:m:s:t:", long_options, &opt_index)) != -1)
    switch (c) {
      case 'h':
        usage(argv[0]);
        return 0;
      case 'i':
        if (optarg[0] == 'c' || optarg[0] == 'C')
          haste_flag = false;
        break;
      case 'm':
        if (optarg[0] == 'i' || optarg[0] == 'I')
          inference_flag = true;
        break;
      case 's':
        sscanf(optarg, "%d", &sample_size);
        break;
      case 't':
        sscanf(optarg, "%d", &time_steps);
        break;
    }

  printf("# Benchmark configuration:\n");
  printf("#   Mode: %s\n", inference_flag ? "inference" : "training");
  printf("#   Implementation: %s\n", haste_flag ? "Haste" : "cuDNN");
  printf("#   Sample size: %d\n", sample_size);
  printf("#   Time steps: %d\n", time_steps);
  printf("#\n");
  printf("# batch_size,hidden_size,input_size,time_ms\n");

  for (const int N : { 1, 16, 32, 64, 128 }) {
    for (const int H : { 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096 }) {
      for (const int C : { 64, 128, 256, 512 }) {
        Tensor2 W(H * 3, C);
        Tensor2 R(H * 3, H);
        Tensor1 bx(H * 3);
        Tensor1 br(H * 3);
        Tensor3 x(C, N, time_steps);
        Tensor3 dh(H, N, time_steps + 1);

        float ms;
        if (inference_flag) {
          if (haste_flag)
            ms = HasteInference(sample_size, W, R, bx, br, x);
          else
            ms = CudnnInference(sample_size, W, R, bx, br, x);
        } else {
          if (haste_flag)
            ms = HasteTrain(sample_size, W, R, bx, br, x, dh);
          else
            ms = CudnnTrain(sample_size, W, R, bx, br, x, dh);
        }
        printf("%d,%d,%d,%f\n", N, H, C, ms);
      }
    }
  }

  cublasDestroy(g_blas_handle);
  cudnnDestroy(g_cudnn_handle);
  return 0;
}

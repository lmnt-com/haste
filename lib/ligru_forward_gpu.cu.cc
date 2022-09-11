
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "blas.h"
#include "device_assert.h"
#include "haste.h"
#include "inline_ops.h"

namespace {


template<typename T, bool Training>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const T* wx,
                         const T* uh,
                         const T* h,
                         T* h_out,
                         T* v,
                         const T* drop_mask) {  
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_dim || col >= batch_dim)
    return;

  const int weight_idx = col * (hidden_dim * 2) + row;

  // Index into the `h` and `h_out` vectors (they have a stride of `hidden_dim`).
  const int output_idx = col * hidden_dim + row;

  // Indicies into the Wx and Rh matrices (for each of the u, r, and e components).
  const int a_idx = weight_idx + 0 * hidden_dim;
  const int z_idx = weight_idx + 1 * hidden_dim;


  const T z = sigmoid(wx[z_idx] + uh[z_idx]);// + uh[z_idx] //wx[z_idx] + uh[z_idx]; // sigmoid(wx[z_idx] + uh[z_idx]);
  const T a = wx[a_idx] + uh[a_idx]; // + uh[a_idx] //+ uh[a_idx];
  
  const T hcand = relu(a); // drop_mask[output_idx];

  // Store internal activations if we're eventually going to backprop.
  if (Training) {
    const int base_v_idx = col * (hidden_dim * 3) + row;
    v[base_v_idx + 1 * hidden_dim] = z;
    v[base_v_idx + 0 * hidden_dim] = a;
    v[base_v_idx + 2 * hidden_dim] = hcand;
  }

  T cur_h_value = z * h[output_idx] + (static_cast<T>(1.0)- z) * hcand;

  h_out[output_idx] = cur_h_value;
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
template<typename T, bool Training>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const half* wx,
                         const half* uh,
                         const half* h,
                         half* h_out,
                         half* v,
                         const half* drop_mask) {
  device_assert_fail("FP16 is not supported on compute capability < 7.0.");
}
#endif

}  // anonymous namespace

namespace haste {
namespace v0 {
namespace ligru {

template<typename T>
struct ForwardPass<T>::private_data {
  bool training;
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
  cudaStream_t sync_stream;
};

template<typename T>
ForwardPass<T>::ForwardPass(
    const bool training,
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
  data_->training = training;
  data_->batch_size = batch_size;
  data_->input_size = input_size;
  data_->hidden_size = hidden_size;
  data_->blas_handle = blas_handle;
  data_->sync_stream = stream;
  cudaStreamCreate(&data_->stream[0]);
  cudaStreamCreate(&data_->stream[1]);
  cudaEventCreateWithFlags(&data_->event, cudaEventDisableTiming);
}

template<typename T>
ForwardPass<T>::~ForwardPass() {
  if (data_->sync_stream) {
    cudaEventRecord(data_->event, data_->stream[1]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
    cudaEventRecord(data_->event, data_->stream[0]);
    cudaStreamWaitEvent(data_->sync_stream, data_->event, 0);
  } else {
    cudaStreamSynchronize(data_->stream[1]);
    cudaStreamSynchronize(data_->stream[0]);
  }
  cudaEventDestroy(data_->event);
  cudaStreamDestroy(data_->stream[1]);
  cudaStreamDestroy(data_->stream[0]);
  delete data_;
}


template<typename T>
void ForwardPass<T>::IterateInternal(
    const T* u, 
    const T* h, 
    T* h_out,  
    T* v,      
    T* tmp_wx,  
    T* tmp_uh,   
    const T* drop_mask) {
    static const T alpha = static_cast<T>(1.0);
    static const T beta = static_cast<T>(0.0);

    const bool training = data_->training;
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;


    cublasSetStream(blas_handle, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size * 2, batch_size, hidden_size,
        &alpha,
        u, hidden_size * 2,
        h, hidden_size,
        &beta,
        tmp_uh, hidden_size * 2);

    // Compute launch configuration for pointwise operations kernel.
    const dim3 blockDim(32, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    cudaStreamWaitEvent(stream1, event, 0);

    if (training) {
        PointwiseOperations<T, true><<<gridDim, blockDim, 0, stream1>>>(
          batch_size,
          hidden_size,
          tmp_wx,
          tmp_uh,
          h,
          h_out,
          v,
          drop_mask);
    }
    else {
        PointwiseOperations<T, false><<<gridDim, blockDim, 0, stream1>>>(
          batch_size,
          hidden_size,
          tmp_wx,
          tmp_uh,
          h,
          h_out,
          v,
          drop_mask); 
    }
}


template<typename T>
void ForwardPass<T>::Run(
    const int seq_length,
    T* wx,
    const T* u,
    T* h,
    T* v,
    T* tmp_uh,
    const T* drop_mask) {
    
    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;

  const int NH = batch_size * hidden_size;
  // auto i = seq_length;

  for (int i = 0; i < seq_length; ++i) {
    IterateInternal
    (
        u, 
        h + i * NH,
        h + (i + 1) * NH,  
        v + i * NH * 3,      
        wx + i * NH * 2,  
        tmp_uh,   
        drop_mask);

  }
}


template struct ForwardPass<half>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;

}
}
}
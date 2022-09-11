#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "blas.h"
#include "device_assert.h"
#include "haste.h"
#include "inline_ops.h"



namespace {


template<typename T>
__global__
void PointwiseOperations(const int batch_dim,
                         const int hidden_dim,
                         const T* h,
                         const T* v,
                         T* dh_prev, 
                         const T* grad_out,
                         T* dwx,
                         const T* drop_mask) {  // Zoneout mask (only used if ApplyZoneout==true)
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row >= hidden_dim || col >= batch_dim)
    return;

  const int base_idx = col * hidden_dim + row;

  T dh = grad_out[base_idx] + dh_prev[base_idx];

  const int stride3_base_idx = col * (hidden_dim * 3) + row;
  const int z_idx = stride3_base_idx + 1 * hidden_dim;
  const int a_idx = stride3_base_idx + 0 * hidden_dim;
  const int hcand_idx = stride3_base_idx + 2 * hidden_dim;

  const T z = v[z_idx];
  const T a = v[a_idx];
  const T hcand = v[hcand_idx];

  // const T tmp = (static_cast<T>(1.0) - z) * dh;
  const T dat = d_relu(a) * (static_cast<T>(1.0) - z) * dh;
  const T dzt = (h[base_idx] - hcand) * dh * (z * (static_cast<T>(1.0) - z));


  dh_prev[base_idx] = dh * z; 

  const int idx = col * (hidden_dim * 2) + row;
  dwx[idx + 1 * hidden_dim] = dzt;//dat; 
  dwx[idx + 0 * hidden_dim] = dat ;
}

}  // anonymous namespace


namespace haste {
namespace v0 {
namespace ligru_v2 {

template<typename T>
struct BackwardPass<T>::private_data {
  int batch_size;
  int input_size;
  int hidden_size;
  cublasHandle_t blas_handle;
  cudaStream_t stream[2];
  cudaEvent_t event;
  cudaStream_t sync_stream;
};


template<typename T>
BackwardPass<T>::BackwardPass(
    const int batch_size,
    const int input_size,
    const int hidden_size,
    const cublasHandle_t& blas_handle,
    const cudaStream_t& stream) : data_(new private_data) {
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
BackwardPass<T>::~BackwardPass() {
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
void BackwardPass<T>::IterateInternal(
    const T* u_t,
    const T* h,
    const T* v,
    const T* grad_out,
    T* dh,
    T* tmp_dwx,
    T* dwx,
    layer_norm::BackwardPass<T>& layer_norm1,
    const T* drop_mask)
{
    const T alpha = static_cast<T>(1.0);
    const T beta_sum = static_cast<T>(1.0);

    const int batch_size = data_->batch_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaEvent_t event = data_->event;

    const dim3 blockDim(32, 16);
    const dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (batch_size + blockDim.y - 1) / blockDim.y);

    PointwiseOperations<T><<<gridDim, blockDim, 0, stream1>>>(
        batch_size,
        hidden_size,
        h,
        v,
        dh, 
        grad_out,
        dwx,
        drop_mask
    );
    cudaEventRecord(event, stream1);
    cublasSetStream(blas_handle,  stream1);
    layer_norm1.RunPartial(stream1, batch_size, dwx, tmp_dwx);


    cudaStreamWaitEvent(stream1, event, 0);
    
    cudaEventRecord(event, stream1);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        hidden_size, batch_size, hidden_size * 2,
        &alpha,
        u_t, hidden_size ,
        tmp_dwx, hidden_size * 2,
        &beta_sum,
        dh, hidden_size);
    cudaStreamWaitEvent(stream1, event, 0);
};

template<typename T>
void BackwardPass<T>::Run(
    const int time_step,
    const T* wx_t,
    const T* u_t,
    const T* h,
    const T* v,
    const T* grad_out,
    T* tmp_dwx,
    T* dwx,
    T* du,
    T* dh,
    layer_norm::BackwardPass<T>& layer_norm1,
    const T* drop_mask) {
    
    const T alpha = static_cast<T>(1.0);
    const T beta_sum = static_cast<T>(1.0);
    const T beta_assign = static_cast<T>(0.0);

    const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

    const int batch_size = data_->batch_size;
    const int input_size = data_->input_size;
    const int hidden_size = data_->hidden_size;
    const cublasHandle_t blas_handle = data_->blas_handle;
    const cudaStream_t stream1 = data_->stream[0];
    const cudaStream_t stream2 = data_->stream[1];
    const cudaEvent_t event = data_->event;

    cudaStream_t save_stream;
    cublasGetStream(blas_handle, &save_stream);

    const int NH = batch_size * hidden_size;
    for (int i = time_step - 1; i >= 0; --i) {
    IterateInternal(
        u_t,
        h + i * NH,
        v + i * NH * 3,
        grad_out + (i + 1) * NH,
        dh,
        tmp_dwx + i * NH * 2,
        dwx + i * NH * 2,
        layer_norm1,
        drop_mask);
    }

    cudaStreamWaitEvent(stream2, event, 0);
  
    cublasSetStream(blas_handle, stream2);
    blas<T>::gemm(blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        hidden_size * 2, hidden_size, batch_size * time_step,
        &alpha,
        tmp_dwx, hidden_size * 2,
        h, hidden_size,
        &beta_sum,
        du, hidden_size * 2);
  cublasSetStream(blas_handle, save_stream);

}

template struct BackwardPass<float>;
template struct BackwardPass<double>;
}
}
}
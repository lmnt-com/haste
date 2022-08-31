#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include "blas.h"
#include "device_assert.h"
#include "haste.h"
#include "inline_ops.h"


// todo: backward of ligru 
// todo: try a side version

namespace haste {
namespace v0 {
namespace ligru {

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
void BackwardPass<T>::Run(
    const int time_step,
    const T* wx,
    const T* u_t,
    const T* h,
    const T* v,
    const T* grad_out,
    T* dwx,
    T* du,
    T* dh,
    const T* drop_mask) {

  const blas<void>::enable_tensor_cores scoped0(data_->blas_handle);
  const blas<void>::set_pointer_mode scoped1(data_->blas_handle);

  // const T alpha = static_cast<T>(1.0);
  // const T beta_sum = static_cast<T>(1.0);
  // const T beta_assign = static_cast<T>(0.0);

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
  for (int i = steps - 1; i >= 0; --i) {
  //   IterateInternal(
  //       R_t,
  //       h + i * NH,
  //       v + i * NH * 4,
  //       dh_new + (i + 1) * NH,
  //       dbx,
  //       dbr,
  //       dh,
  //       dp + i * NH * 3,
  //       dq + i * NH * 3,
  //       zoneout_mask ? zoneout_mask + i * NH : nullptr );
  }

}


template struct BackwardPass<half>;
template struct BackwardPass<float>;
template struct BackwardPass<double>;
}
}
}
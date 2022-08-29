
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include "blas.h"
#include "device_assert.h"
#include "haste.h"
#include "inline_ops.h"


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

template struct ForwardPass<half>;
template struct ForwardPass<float>;
template struct ForwardPass<double>;

}
}
}
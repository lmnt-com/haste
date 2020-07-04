AR ?= ar
CXX ?= g++
NVCC ?= nvcc -ccbin $(CXX)
PYTHON ?= python

ifeq ($(OS),Windows_NT)
LIBHASTE := haste.lib
CUDA_HOME ?= $(CUDA_PATH)
AR := lib
AR_FLAGS := /nologo /out:$(LIBHASTE)
NVCC_FLAGS := -x cu -Xcompiler "/MD"
else
LIBHASTE := libhaste.a
CUDA_HOME ?= /usr/local/cuda
AR ?= ar
AR_FLAGS := -crv $(LIBHASTE)
NVCC_FLAGS := -std=c++11 -x cu -Xcompiler -fPIC
endif

LOCAL_CFLAGS := -I/usr/include/eigen3 -I$(CUDA_HOME)/include -Ilib -O3
LOCAL_LDFLAGS := -L$(CUDA_HOME)/lib64 -L. -lcudart -lcublas
GPU_ARCH_FLAGS := -gencode arch=compute_37,code=compute_37 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_70,code=compute_70

# Small enough project that we can just recompile all the time.
.PHONY: all haste haste_tf haste_pytorch libhaste_tf examples benchmarks clean

all: haste haste_tf haste_pytorch examples benchmarks

haste:
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/lstm_forward_gpu.cu.cc -o lib/lstm_forward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/lstm_backward_gpu.cu.cc -o lib/lstm_backward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/gru_forward_gpu.cu.cc -o lib/gru_forward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/gru_backward_gpu.cu.cc -o lib/gru_backward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/layer_norm_forward_gpu.cu.cc -o lib/layer_norm_forward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/layer_norm_backward_gpu.cu.cc -o lib/layer_norm_backward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/layer_norm_lstm_forward_gpu.cu.cc -o lib/layer_norm_lstm_forward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/layer_norm_lstm_backward_gpu.cu.cc -o lib/layer_norm_lstm_backward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/layer_norm_gru_forward_gpu.cu.cc -o lib/layer_norm_gru_forward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/layer_norm_gru_backward_gpu.cu.cc -o lib/layer_norm_gru_backward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/indrnn_backward_gpu.cu.cc -o lib/indrnn_backward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/indrnn_forward_gpu.cu.cc -o lib/indrnn_forward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/layer_norm_indrnn_forward_gpu.cu.cc -o lib/layer_norm_indrnn_forward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/layer_norm_indrnn_backward_gpu.cu.cc -o lib/layer_norm_indrnn_backward_gpu.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(AR) $(AR_FLAGS) lib/*.o

libhaste_tf: haste
	$(eval TF_CFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
	$(eval TF_LDFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))
	$(CXX) -std=c++11 -c frameworks/tf/lstm.cc -o frameworks/tf/lstm.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/gru.cc -o frameworks/tf/gru.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/layer_norm.cc -o frameworks/tf/layer_norm.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/layer_norm_gru.cc -o frameworks/tf/layer_norm_gru.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/layer_norm_indrnn.cc -o frameworks/tf/layer_norm_indrnn.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/layer_norm_lstm.cc -o frameworks/tf/layer_norm_lstm.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/indrnn.cc -o frameworks/tf/indrnn.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -std=c++11 -c frameworks/tf/support.cc -o frameworks/tf/support.o $(LOCAL_CFLAGS) $(TF_CFLAGS) -fPIC
	$(CXX) -shared frameworks/tf/*.o libhaste.a -o frameworks/tf/libhaste_tf.so $(LOCAL_LDFLAGS) $(TF_LDFLAGS) -fPIC

# Dependencies handled by setup.py
haste_tf:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cat build/common.py build/setup.tf.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q bdist_wheel)
	@cp $(TMP)/dist/*.whl .
	@rm -rf $(TMP)

# Dependencies handled by setup.py
haste_pytorch:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cat build/common.py build/setup.pytorch.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q bdist_wheel)
	@cp $(TMP)/dist/*.whl .
	@rm -rf $(TMP)

dist:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cp build/MANIFEST.in $(TMP)
	@cat build/common.py build/setup.tf.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q sdist)
	@cp $(TMP)/dist/*.tar.gz .
	@rm -rf $(TMP)
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cp build/MANIFEST.in $(TMP)
	@cat build/common.py build/setup.pytorch.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q sdist)
	@cp $(TMP)/dist/*.tar.gz .
	@rm -rf $(TMP)

examples: haste
	$(CXX) -std=c++11 examples/lstm.cc $(LIBHASTE) $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o haste_lstm -Wno-ignored-attributes
	$(CXX) -std=c++11 examples/gru.cc $(LIBHASTE) $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o haste_gru -Wno-ignored-attributes

benchmarks: haste
	$(CXX) -std=c++11 benchmarks/benchmark_lstm.cc $(LIBHASTE) $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o benchmark_lstm -Wno-ignored-attributes -lcudnn
	$(CXX) -std=c++11 benchmarks/benchmark_gru.cc $(LIBHASTE) $(LOCAL_CFLAGS) $(LOCAL_LDFLAGS) -o benchmark_gru -Wno-ignored-attributes -lcudnn

clean:
	rm -fr benchmark_lstm benchmark_gru haste_lstm haste_gru haste_*.whl haste_*.tar.gz
	find . \( -iname '*.o' -o -iname '*.so' -o -iname '*.a' -o -iname '*.lib' \) -delete

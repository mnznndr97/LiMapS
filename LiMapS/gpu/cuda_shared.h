#pragma once

#include <assert.h>
#include <memory>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(call)                                                                \
{                                                                                       \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess)                                                           \
    {                                                                                   \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                          \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));    \
        assert(false);                                                                  \
    }                                                                                   \
}

#define CUDA_CHECKD(call)                                                                \
{                                                                                       \
    const cudaError_t error = call;                                                     \
    if (error != cudaSuccess)                                                           \
    {                                                                                   \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                                   \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));             \
        assert(false);                                                                  \
    }                                                                                   \
}

template<class T> struct CudaPtrDeleter {
    void operator()(T* ptr) const {
        CUDA_CHECK(cudaFree(ptr));
    }
};

template<typename T>
using cuda_ptr = std::unique_ptr<T, CudaPtrDeleter<T>>;

template<typename T>
using cuda_managed_ptr = std::unique_ptr<T, CudaPtrDeleter<T>>;

template<class T>
cuda_ptr<T> make_cuda(std::size_t count)
{
    T* ptr = NULL;
	CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * count));
	return cuda_ptr<T>(ptr);
}

template<class T>
cuda_managed_ptr<T> make_cuda_managed(std::size_t count)
{
	T* ptr = NULL;
	CUDA_CHECK(cudaMallocManaged<T>(&ptr, sizeof(T) * count, cudaMemAttachGlobal));
	return cuda_managed_ptr<T>(ptr);
}

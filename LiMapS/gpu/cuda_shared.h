#pragma once

#include <assert.h>
#include <memory>

#include "cuda.h"

#ifdef __INTELLISENSE__
#define __CUDACC__
#include "cuda_intrinsics.h"
#endif  

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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

template<class TCallback> struct CudaPtrDeleter {
    void operator()(TCallback* ptr) const {
        CUDA_CHECK(cudaFree(ptr));
    }
};

template<typename TCallback>
using cuda_ptr = std::unique_ptr<TCallback, CudaPtrDeleter<TCallback>>;

template<typename TCallback>
using cuda_managed_ptr = std::unique_ptr<TCallback, CudaPtrDeleter<TCallback>>;

template<class TCallback>
cuda_ptr<TCallback> make_cuda(std::size_t count)
{
    TCallback* ptr = NULL;
	CUDA_CHECK(cudaMalloc(&ptr, sizeof(TCallback) * count));
	return cuda_ptr<TCallback>(ptr);
}

template<class TCallback>
cuda_managed_ptr<TCallback> make_cuda_managed(std::size_t count)
{
	TCallback* ptr = NULL;
	CUDA_CHECK(cudaMallocManaged<TCallback>(&ptr, sizeof(TCallback) * count, cudaMemAttachGlobal));
	return cuda_managed_ptr<TCallback>(ptr);
}

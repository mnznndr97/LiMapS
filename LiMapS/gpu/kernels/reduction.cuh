#pragma once

#include "../cuda_shared.h"
#include <nvfunctional>

#define FULL_MASK 0xffffffff

/// <summary>
/// Sums all the data relatively to a warp
/// </summary>
/// <param name="data">Warp data</param>
/// <returns>Sums of the data provided bu the threads in the warp</returns>
__inline__ __device__ float WarpReduce(float data) {
	// Each warp thread sums its own value with the +16 thread, +8 thread, ecc
	// In this way we sum without race conditions our data at warp level
	data += __shfl_xor_sync(FULL_MASK, data, 16);
	data += __shfl_xor_sync(FULL_MASK, data, 8);
	data += __shfl_xor_sync(FULL_MASK, data, 4);
	data += __shfl_xor_sync(FULL_MASK, data, 2);
	data += __shfl_xor_sync(FULL_MASK, data, 1);
	return data;
}

/// <summary>
/// Kernel reduce function that uses local memory to store the <paramref name="sumCallback"/> parameter
/// </summary>
template <typename... Arguments>
__device__ void KernelReduce_LocalMemory(float data, size_t size, const nvstd::function<void(Arguments..., float)>&& sumCallback, Arguments... args) {
	extern __shared__ float blockParSums[];

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	// After this first step, in the shared mem we have the partials sums for the block subdivided by warp size
	float warpSum = WarpReduce(data);
	if (warpLane == 0) {
		// NB: if we are in the "out-of-bounds" region of the data, sum should be zero .
		// This is necessary to have a legitimate value in the shared mem that later will 
		// be used in another reduce pass
		blockParSums[warpIndex] = warpSum;
	}

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 
	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;

	// Threads int the first warp will now sum all the partial data, and the first thread in block will invoke the callback
	// At the end, we report a partial sum for each block to the calling function
	float blockSum = warpIndex == 0 ? WarpReduce(warpParSum) : 0.0f;
	if (threadIdx.x == 0) {
		sumCallback(args..., blockSum);
	}
}

/// <summary>
/// "Optimized" kernel reduce function that uses local memory to store the <paramref name="sumCallback"/> parameter
/// </summary>
template <typename TCallback, typename... Arguments>
__device__ void KernelReduce(float data, TCallback sumCallback, Arguments... args) {
	extern __shared__ float blockParSums[];

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	// After this first step, in the shared mem we have the partials sums for the block subdivided by warp size
	float warpSum = WarpReduce(data);
	if (warpLane == 0) {
		// NB: if we are in the "out-of-bounds" region of the data, sum should be zero .
		// This is necessary to have a legitimate value in the shared mem that later will 
		// be used in another reduce pass
		blockParSums[warpIndex] = warpSum;
	}

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 
	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;

	// Threads int the first warp will now sum all the partial data, and the first thread in block will invoke the callback
	// At the end, we report a partial sum for each block to the calling function
	float blockSum = WarpReduce(warpParSum);
	if (threadIdx.x == 0) {
		sumCallback(args..., blockSum);
	}
}
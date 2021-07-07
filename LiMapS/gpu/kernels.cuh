#pragma once

#include "cuda_shared.h"

#define FULL_MASK 0xffffffff

/// <summary>
/// Sums all the data relatively to a warp
/// </summary>
/// <param name="data">Warp data</param>
/// <returns>Sums of the </returns>
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

__device__ void Norm(const float* vec, size_t size, float* result);
__global__ void SquareSumKrnl(const float* vec, size_t size, float* result);
__device__ void NormDiff(const float* vec1, const float* vec2, size_t size, float* result);
__global__ void NormDiffKrn(const float* vec1, const float* vec2, size_t size, float* result);


template <int unrollFactor>
__global__ void SquareSumKrnlUnroll(const float* vec, size_t size, float* result) {

	extern __shared__ float blockParSums[];

	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums

	float data = 0.0f;
#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		float d = (idx + i * blockDim.x) < size ? vec[idx + i * blockDim.x] : 0.0f;
		data += (d * d);
	}

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	float warpSum = WarpReduce(data);
	if (warpLane == 0) {
		// NB: if we are in the "out-of-bounds" region of the data, sum shound be zero 
		// This is necessary to have a legitimate value in the shared mem that later will 
		// be used in another reduce pass
		assert(warpSum >= 0.0f);
		blockParSums[warpIndex] = warpSum;
	}

	if (idx >= size) return;

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 

	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;
	assert(warpParSum >= 0.0f);

	// Facciamo lavorare solo il primo warp
	float blockSum = warpIndex == 0 ? WarpReduce(warpParSum) : 0.0f;
	assert(blockSum >= 0.0f);

	if (threadIdx.x == 0) {
		atomicAdd(result, blockSum);
	}
}

template <int unrollFactor>
__global__ void SquareSumKrnlUnrollLdg(const float* vec, size_t size, float* result) {
	extern __shared__ float blockParSums[];

	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums

	float data = 0.0f;
#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		float d = (idx + i * blockDim.x) < size ? __ldg(&vec[idx + i * blockDim.x]) : 0.0f;
		data += (d * d);
	}

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	float warpSum = WarpReduce(data);
	if (warpLane == 0) {
		// NB: if we are in the "out-of-bounds" region of the data, sum shound be zero 
		// This is necessary to have a legitimate value in the shared mem that later will 
		// be used in another reduce pass
		assert(warpSum >= 0.0f);
		blockParSums[warpIndex] = warpSum;
	}

	//if (idx >= size) return;

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 

	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;
	assert(warpParSum >= 0.0f);

	// Facciamo lavorare solo il primo warp
	float blockSum = warpIndex == 0 ? WarpReduce(warpParSum) : 0.0f;
	assert(blockSum >= 0.0f);

	if (threadIdx.x == 0) {
		atomicAdd(result, blockSum);
	}
}
#pragma once

#include "cuda_shared.h"
#include <nvfunctional>

#define FULL_MASK 0xffffffff

__inline__ __device__ float GetBeta(float lambda, float data) {
	return data * (1.0f - expf(-lambda * fabs(data)));
}

__global__ void GetBetaKrnl(float lambda, const float* data, float* beta, size_t size);

__inline__ __host__ __device__ dim3 GetGridSize(const dim3& blockSize, size_t dataSize, int unrollingFactor) {
	assert(dataSize > 0);
	assert(unrollingFactor > 0);

	// NB: We just set the x dimension here, but for the y and z should be the same
	// In our application only one dimension is needed

	// We first fix the grid size using the block dimension
	dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);

	// Then we account also for unrolling factor, since our data length might not be a power of two
	gridSize.x = (gridSize.x + unrollingFactor - 1) / unrollingFactor;

	assert((gridSize.x * unrollingFactor * blockSize.x) >= dataSize);
	assert(gridSize.y == 1);
	assert(gridSize.z == 1);
	return gridSize;
}

__inline__ __host__ __device__ dim3 GetGridSize2(const dim3& blockSize, size_t dataSize, int unrollingFactor, size_t dataSize2, int unrollingFactor2) {
	assert(dataSize > 0);
	assert(unrollingFactor > 0);

	// NB: We just set the x dimension here, but for the y and z should be the same
	// In our application only one dimension is needed

	// We first fix the grid size using the block dimension
	dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x, (dataSize2 + blockSize.y - 1) / blockSize.y);

	// Then we account also for unrolling factor, since our data length might not be a power of two
	gridSize.x = (gridSize.x + unrollingFactor - 1) / unrollingFactor;
	gridSize.y = (gridSize.y + unrollingFactor2 - 1) / unrollingFactor2;

	assert((gridSize.x * unrollingFactor * blockSize.x) >= dataSize);
	assert((gridSize.y * unrollingFactor2 * blockSize.y) >= dataSize2);
	assert(gridSize.z == 1);
	return gridSize;
}

__inline__ __host__ __device__ dim3 GetGridSize(const dim3& blockSize, size_t dataSize) {
	return GetGridSize(blockSize, dataSize, 1);
}


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

template <typename... Arguments>
__device__ void KernelReduce(float data, size_t size, nvstd::function<void(Arguments..., float)> sumCallback, Arguments... args) {
	extern __shared__ float blockParSums[];

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	float warpSum = WarpReduce(data);
	if (warpLane == 0) {
		// NB: if we are in the "out-of-bounds" region of the data, sum shound be zero 
		// This is necessary to have a legitimate value in the shared mem that later will 
		// be used in another reduce pass
		blockParSums[warpIndex] = warpSum;
	}

	if (idx >= size) return;

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 

	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;

	// Facciamo lavorare solo il primo warp
	float blockSum = warpIndex == 0 ? WarpReduce(warpParSum) : 0.0f;

	// AlphaOld and Alpha should be aligned at the first iteration, so let's write them directly here
	// This avoids us a vector copy later
	if (threadIdx.x == 0) {
		sumCallback(args..., blockSum);
	}
}


__global__ void SquareSumKrnl(const float* vec, size_t size, float* result);

template <int unrollFactor>
__global__ void SquareSumKrnlUnroll(const float* vec, size_t size, float* result) {
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

	KernelReduce<float*>(data, size, [](float* result, float data) {
		atomicAdd(result, data);
		}, result);
}

template <int unrollFactor>
__global__ void SquareSumKrnlUnrollLdg(const float* vec, size_t size, float* result) {
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

	KernelReduce<float*>(data, size, [](float* result, float data) {
		atomicAdd(result, data);
		}, result);
}

template <int unrollFactor>
__global__ void SquareDiffSumKrnlUnroll(const float* vec1, const float* vec2, size_t size, float* result) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums

	float data = 0.0f;
#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float d = vOffset < size ? vec1[vOffset] - vec2[vOffset] : 0.0f;
		data += (d * d);
	}

	KernelReduce<float*>(data, size, [](float* result, float data) {
		atomicAdd(result, data);
		}, result);
}
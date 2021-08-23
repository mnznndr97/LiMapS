#pragma once

#include "..\cuda_shared.h"
#include <nvfunctional>

#define FULL_MASK 0xffffffff



__inline__ __device__ float GetBeta(float lambda, float data)
{
	return data * (1.0f - expf(-lambda * fabs(data)));
}

/// <summary>
/// Copies a vector of floats to another, optionally negating the values
/// </summary>
template<int unrollFactor>
__global__ void CopyTo(const float* source, size_t size, float* dest, bool negate) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;
	// MicroOpt: this may be passed as template parameter
	float a = negate ? -1.0f : 1.0f;

#pragma unroll
	for (int i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		if (vOffset < size) dest[vOffset] = a * source[vOffset];
	}
}

template<int unrollFactor>
__global__ void CalculateBeta(const float* __restrict__ alpha, float* beta, float lambda, size_t dictionaryWords) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

#pragma unroll
	for (int i = 0; i < unrollFactor; i++) {
		size_t index = i * blockDim.x + idx;

		if (index < dictionaryWords)
			beta[index] = GetBeta(lambda, alpha[index]);
	}
}

/// <summary>
/// Calculates the necessary grid x dimension considering the data size and the unrolling factor
/// </summary>
__inline__ __host__ __device__ dim3 GetGridSize(const dim3& blockSize, size_t dataSize, int unrollingFactor) {
	assert(dataSize > 0);
	assert(unrollingFactor > 0);

	// NB: We just set the x dimension here, but for the y and z should be the same
	// In our application only one dimension is needed

	// We first fix the grid size using the block dimension
	dim3 gridSize((unsigned int)((dataSize + blockSize.x - 1) / blockSize.x));

	// Then we account also for unrolling factor, since our data length might not be a power of two
	gridSize.x = (gridSize.x + unrollingFactor - 1) / unrollingFactor;

	assert((gridSize.x * unrollingFactor * blockSize.x) >= dataSize);
	assert(gridSize.y == 1);
	assert(gridSize.z == 1);
	return gridSize;
}

__inline__ __host__ __device__ dim3 GetGridSize(const dim3& blockSize, size_t dataSize) {
	return GetGridSize(blockSize, dataSize, 1);
}



#pragma once

#include "cuda_shared.h"
#include <nvfunctional>

#define FULL_MASK 0xffffffff
#define TILE_DIM    16
#define BLOCK_ROWS  16

__inline__ __device__ size_t Mod(size_t dividend, size_t divisor) {
	while (dividend >= divisor)
		dividend -= divisor;
	return dividend;
}

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

template<typename TCallback, typename... Arguments>
__global__ void Matrix2VectorPartition(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height, TCallback sumCallback, Arguments... args) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0f;
	// We iterate over block size to reach width. We iterate in blocks of a size of a warp
	// In this way vector items are read one time in a single transaction for each warp
	// NB: each thread is executing the operations, even if it is operating with an out-of-bound index;
	// in this case the value calculated will always be 0.0f
	for (size_t i = 0; i < ((width + warpSize - 1) / warpSize); i++)
	{
		size_t base = i * warpSize;
		size_t index = base + (threadIdx.x % warpSize);

		// Vector read: each thread in the warp reads it own items, contiguous to the others
		float data = 0.0f;
		if (index < width) {
			index = (index + (blockDim.x * blockIdx.x));
			index = Mod(index, width);
			data = vector[index];
		}

		// After data read, each thread in the warp shares the vector value with the other threads to calculare the sum
#pragma unroll
		for (int w = 0; w < warpSize; w++)
		{
			float vectorData = __shfl_sync(FULL_MASK, data, w);
			float matrixData = 0.0f;
			// Each thread in the warp is reading the same item but in different rows. So better use a column-major ordering
			if ((idx < height) && (base + w < width)) {
				index = base + (blockDim.x * blockIdx.x) + w;
				index = Mod(index, width);
				index = index * height + idx;
				matrixData = matrix[index];
			}
			sum += matrixData * vectorData;
		}
	}

	// Now if we are in our bound, we invoke the callback
	if (idx < height) {
		sumCallback(args..., dest, idx, sum);
	}
}

__inline__ __global__ void Matrix2VectorStream(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height, size_t indexOffset) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	idx += indexOffset;

	float sum = 0.0f;
	// We iterate over block size to reach width. We iterate in blocks of a size of a warp
	// In this way vector items are read one time in a single transaction for each warp
	// NB: each thread is executing the operations, even if it is operating with an out-of-bound index;
	// in this case the value calculated will always be 0.0f
	for (size_t i = 0; i < ((width + warpSize - 1) / warpSize); i++)
	{
		size_t base = i * warpSize;
		size_t index = base + (threadIdx.x % warpSize);

		// Vector read: each thread in the warp reads it own items, contiguous to the others
		float data = index < width ? vector[index] : 0.0f;

		// After data read, each thread in the warp shares the vector value with the other threads to calculare the sum
#pragma unroll
		for (int w = 0; w < warpSize; w++)
		{
			float vectorData = __shfl_sync(FULL_MASK, data, w);
			// Each thread in the warp is reading the same item but in different rows. So better use a column-major ordering
			float matrixData = ((idx < height) && ((base + w) < width)) ? matrix[(base + w) * height + idx] : 0.0f;
			sum += matrixData * vectorData;
		}
	}

	// Now if we are in our bound, we invoke the callback
	if (idx < height) {
		dest[idx] = sum;
	}
}


__global__ void Matrix2VectorSharedMem(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height);

/// <summary>
/// Applies a threshold to a vector. All the elements below the threshold are zeroed
/// </summary>
template<int unrollFactor>
__global__ void ThresholdVector(float* vector, size_t size) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		if (vOffset < size) {
			// Little optimization here: if the data is already zero, we can avoid a memory write. This may
			// improve the performance in the final stages of the algorithm where most of the solution elements are 0
			float data = vector[vOffset];

			// NB: Threshold may be a parameter read by constant memory here
			// For our application, an hard coded may be an optimization since the value
			// should not change "frequently" depending on the application
			if (data != 0.0f && fabs(data) < 1e-4f)
				vector[vOffset] = 0.0f;
		}
	}
}

/// <summary>
/// Applies a threshold to a vector. All the elements below the threshold are zeroed
/// </summary>
template<int unrollFactor>
__global__ void ThresholdVectorAlwaysWrite(float* vector, size_t size) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		if (vOffset < size) {
			float data = vector[vOffset];
			if (fabs(data) < 1e-4f)
				vector[vOffset] = 0.0f;
		}
	}
}

/// <summary>
/// Transpose that effectively reorders execution of thread blocks along diagonals of the
/// matrix (also coalesced and has no bank conflicts)
///
/// Here blockIdx.x is interpreted as the distance along a diagonal and blockIdx.y as
/// corresponding to different diagonals
///
/// blockIdx_x and blockIdx_y expressions map the diagonal coordinates to the more commonly
/// used cartesian coordinates so that the only changes to the code from the coalesced version
/// are the calculation of the blockIdx_x and blockIdx_y and replacement of blockIdx.x and
/// bloclIdx.y with the subscripted versions in the remaining code
/// </summary>
__global__ void Transpose(const float* __restrict__ source, float* destination, size_t width, size_t height);
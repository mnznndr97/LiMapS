#pragma once

#include "cuda_shared.h"
#include <nvfunctional>

#define FULL_MASK 0xffffffff

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

__inline__ __device__ float GetBeta(float lambda, float data) {
	return data * (1.0f - expf(-lambda * fabs(data)));
}

__global__ void GetBetaKrnl(float lambda, const float* data, float* beta, size_t size);

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

template <typename... Arguments>
__device__ void KernelReduce(float data, size_t size, nvstd::function<void(Arguments..., float)> sumCallback, Arguments... args) {
	extern __shared__ float blockParSums[];

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

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

	if (idx >= size) return;
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
/// Performs a matrix to vector moltiplication
/// </summary>
template<int unrollFactor>
__global__ void Matrix2Vector(const float* matrix, const float* vector, float* dest, size_t width, size_t height, bool negate) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;

	// We calculate our destination (and source) row index. If it is out of bound, we can immediatly exit
	if (row >= height) return;

	// We now have to perform a single (also considering unrolling) row * col item multiplication
	float alpha = negate ? -1.0f : 1.0f;
	float data = 0.0f;
#pragma unroll
	for (int i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float mat = vOffset < width ? matrix[row * width + vOffset] : 0.0f;
		float vec = vOffset < width ? vector[vOffset] : 0.0f;

		data += (mat * vec);
		// There is the MultAdd intrinsic but it seems to make no difference, nor in debug mode nor in release
		// Maybe the compiler it's already efficient in generating this code
		//data = fmaf(mat, vec, data);
	}

	// After the multiplication, each thread will hold it's own sum, and we can apply our beloved reduction
	KernelReduce<float*, float>(data, width, [](float* ptr, float a, float sum) {
		atomicAdd(ptr, a * sum);
		}, &dest[row], alpha);
}

/// <summary>
/// Performs a matrix to vector moltiplication
/// </summary>
template<int unrollFactor>
__global__ void Matrix2Vector(cudaTextureObject_t matrixTex, const float* vector, float* dest, size_t width, size_t height, bool negate) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy >= height) return;

	float alpha = negate ? -1.0f : 1.0f;
	float data = 0.0f;
#pragma unroll
	for (int i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		// With textures, out of bound reads are automatically "handled"
		float mat = tex2D<float>(matrixTex, vOffset, idy);
		float vec = vOffset < width ? vector[vOffset] : 0.0f;

		//data += (dicInv * interm);
		data = fmaf(mat, vec, data);
	}

	// After the multiplication, each thread will hold it's own sum, and we can apply our beloved reduction
	KernelReduce<float*, float>(data, width, [](float* ptr, float a, float sum) {
		atomicAdd(ptr, a * sum);
		}, &dest[idy], alpha);
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

__global__ void SquareSumKrnl(const float* vec, size_t size, float* result);

template <int unrollFactor>
__global__ void SquareSumKrnlUnroll(const float* vec, size_t size, float* result) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums

	float data = 0.0f;
#pragma unroll
	for (int i = 0; i < unrollFactor; i++)
	{
		size_t offset = idx + i * blockDim.x;
		float d = offset < size ? vec[offset] : 0.0f;
		data += (d * d);
	}

	KernelReduce<float*>(data, size, [](float* result, float data) {
		atomicAdd(result, data);
		}, result);
}

template <int unrollFactor>
__global__ void SquareSumGridUnroll(const float* vec, size_t size, float* result) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums

	float data = 0.0f;
	size_t offset = idx;
	while (offset < size) {
#pragma unroll
		for (int i = 0; i < unrollFactor; i++)
		{
			// For our application, most of the access should be aligned and coalesced (we have no by column reads)
			// so readonly cache does not make much difference
			float d = (offset + i * blockDim.x ) < size ? vec[offset + i * blockDim.x] : 0.0f;
			data += (d * d);
		}

		offset += gridDim.x * unrollFactor * blockDim.x;
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
	for (int i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float d = vOffset < size ? vec1[vOffset] - vec2[vOffset] : 0.0f;
		data += (d * d);
	}

	KernelReduce<float*>(data, size, [](float* result, float data) {
		atomicAdd(result, data);
		}, result);
}

template <int unrollFactor>
__global__ void SquareDiffSumKrnlUnrollLdg(const float* vec1, const float* vec2, size_t size, float* result) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums

	float data = 0.0f;
#pragma unroll
	for (int i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float d = vOffset < size ? __ldcs(&vec1[vOffset]) - __ldcs(&vec2[vOffset]) : 0.0f;
		data += (d * d);
	}

	KernelReduce<float*>(data, size, [](float* result, float data) {
		atomicAdd(result, data);
		}, result);
}

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
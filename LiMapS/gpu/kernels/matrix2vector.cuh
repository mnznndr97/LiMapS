#pragma once

#include "../cuda_shared.h"
#include "reduction.cuh"


static __device__ __inline__ void MatAtomicAdd(float* ptr, float a, float sum) {
	atomicAdd(ptr, a * sum);
}

/// <summary>
/// Performs a matrix to vector moltiplication using the kernel reduction to sum the various row*col cells products
/// </summary>
template<int unrollFactor, bool negate>
__global__ void Matrix2Vector(const float* matrix, const float* vector, float* dest, size_t width, size_t height) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;

	// We calculate our destination (and source) row index. If it is out of bound, we can immediatly exit
	if (row >= height) return;

	// We now have to perform a single (also considering unrolling) row * col item multiplication
	float data = 0.0f;

	//size_t offset = idx;
	//while (offset < width) {

#pragma unroll
	for (int i = 0; i < unrollFactor; i++)
	{
		size_t index = idx + i * blockDim.x;

		// Very micro optimization here: we compute only a single branch and we do the fma only
		// if we are in the data bounds.
		if (index < width) {
			data += matrix[row * width + index] * vector[index];
		}

		// There is the MultAdd intrinsic but it seems to make no difference, nor in debug mode nor in release
		// Maybe the compiler it's already efficient in generating this code
		//data = fmaf(mat, vec, data);
	}

	//	offset += gridDim.x * unrollFactor * blockDim.x;
	//}

	// After the multiplication, each thread will hold it's own sum, and we can apply our beloved reduction
	KernelReduce<void(float*, float, float), float*, float>(data, MatAtomicAdd, &dest[row], negate ? -1.0f : 1.0f);
}

/// <summary>
/// Performs a matrix to vector moltiplication using texture
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
	KernelReduce<void(float*, float, float), float*, float>(data, MatAtomicAdd, &dest[idy], alpha);
}

/// <summary>
/// Performs a matrix to vector moltiplication without the support of kernel reduction to avoid too many memory accesses
/// </summary>
template<typename TCallback, typename... Arguments>
__global__ void Matrix2Vector2(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height, TCallback sumCallback, Arguments... args) {
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
		sumCallback(args..., dest, idx, sum);
	}
}

__global__ __inline__ void Matrix2VectorPartitionB(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height) {
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
		dest[idx] = sum;
	}
}
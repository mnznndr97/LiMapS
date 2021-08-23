#pragma once

#include "../cuda_shared.h"
#include "reduction.cuh"

/// <summary>
/// Basic implementation of a vector square-sum vector with no unrolling
/// </summary>
__global__ void SquareSumKrnl(const float* vec, size_t size, float* result);

/// <summary>
/// Implementation of a vector square-sum vector with templated unrolling factor
/// </summary>
template <int unrollFactor>
__global__ void SquareSumKrnlUnroll(const float* __restrict__ vec, size_t size, float* result) {
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

	KernelReduce<float(float*, float), float*>(data, atomicAdd, result);
}

/// <summary>
/// Implementation of a vector square-sum vector with templated unrolling factor and grid-unrolling
/// </summary>
template <int unrollFactor>
__global__ void SquareSumGridUnroll(const float* __restrict__ vec, size_t size, float* result) {
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
			float d = (offset + i * blockDim.x) < size ? vec[offset + i * blockDim.x] : 0.0f;
			data += (d * d);
		}

		offset += gridDim.x * unrollFactor * blockDim.x;
	}

	KernelReduce<float(float*, float), float*>(data, atomicAdd, result);
}

/// <summary>
/// Implementation of a vector difference square-sum vector with templated unrolling factor and grid-unrolling
/// </summary>
template <int unrollFactor>
__global__ void SquareDiffSumKrnlUnroll(const float* __restrict__ vec1, const float* __restrict__ vec2, size_t size, float* result) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

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

	KernelReduce<float(float*, float), float*>(data, atomicAdd, result);
}

template <int unrollFactor>
__global__ void SquareDiffSumKrnlUnrollLdg(const float* vec1, const float* vec2, size_t size, float* result) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

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

	KernelReduce<float(float*, float), float*>(data, atomicAdd, result);
}
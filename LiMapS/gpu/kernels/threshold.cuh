#pragma once

#include "..\cuda_shared.h"
#include <nvfunctional>

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
			// For our application, an hard coded value may be an optimization since the value
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
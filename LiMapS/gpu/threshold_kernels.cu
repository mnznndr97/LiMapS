#include "cuda_shared.h"
#include "threshold_kernels.cuh"

__global__ void ThresholdKrnl(float* data, size_t size, float threshold) {
	size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
	if (offset >= size) return;

	float value = abs(data[offset]);
	if(value < threshold)
		data[offset] = 0.0f;
}
#include "beta_kernels.cuh"
#include "cuda_shared.h"

__global__ void GetBetaKrnl(float lambda, const float* data, float* beta, size_t size) {
	size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
	if (offset >= size) return;

	float alphaValue = data[offset];
	beta[offset] = alphaValue * (1.0f - exp(-lambda * abs(alphaValue)));
}
#include "kernels.cuh"


__global__ void GetBetaKrnl(float lambda, const float* data, float* beta, size_t size) {
	size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
	if (offset >= size) return;

	beta[offset] = GetBeta(lambda, data[offset]);
}


__global__ void SquareSumKrnl(const float* vec, size_t size, float* result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums
	float data = idx < size ? vec[idx] : 0.0f;
	data = data * data;

	KernelReduce<float*>(data, size, [](float* dest, float sum) { atomicAdd(dest, sum); }, result);
}
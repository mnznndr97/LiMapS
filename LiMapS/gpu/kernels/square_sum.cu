#include "square_sum.cuh"

__global__ void SquareSumKrnl(const float* vec, size_t size, float* result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums
	float data = idx < size ? vec[idx] : 0.0f;
	data = data * data;

	KernelReduce<float(float*, float), float*>(data, atomicAdd, result);
}
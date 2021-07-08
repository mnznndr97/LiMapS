#include "kernels.cuh"




__global__ void GetBetaKrnl(float lambda, const float* data, float* beta, size_t size) {
	size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
	if (offset >= size) return;

	beta[offset] = GetBeta(lambda, data[offset]);
}

__global__ void SquareSumKrnl(const float* vec, size_t size, float* result) {
	extern __shared__ float blockParSums[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums
	float data = idx < size ? vec[idx] : 0.0f;
	data = data * data;

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	float warpSum = WarpReduce(data);
	if (warpLane == 0) {
		// NB: if we are in the "out-of-bounds" region of the data, sum shound be zero 
		// This is necessary to have a legitimate value in the shared mem that later will 
		// be used in another reduce pass
		assert(warpSum >= 0.0f);
		blockParSums[warpIndex] = warpSum;
	}

	if (idx >= size) return;

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 

	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;
	assert(warpParSum >= 0.0f);

	// Facciamo lavorare solo il primo warp
	float blockSum = warpIndex == 0 ? WarpReduce(warpParSum) : 0.0f;
	assert(blockSum >= 0.0f);

	if (threadIdx.x == 0) {
		atomicAdd(result, blockSum);
	}
}
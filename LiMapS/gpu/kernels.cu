#include "kernels.cuh"

#define FULL_MASK 0xffffffff

/// <summary>
/// Sums all the data relatively to a warp
/// </summary>
/// <param name="data">Warp data</param>
/// <returns>Sums of the </returns>
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

__global__ void Norm(const float* vec, size_t size, float* result) {
	extern __shared__ float blockParSums[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We calculate the squared value
	float data = vec[idx];
	data = data * data;

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	float warpSum = WarpReduce(data);
	if (warpLane == 0) blockParSums[warpIndex] = warpSum;

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 

	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;
	// Facciamo lavorare solo il primo warp
	float blockSum = warpIndex == 0 ? WarpReduce(warpParSum) : 0.0f;

	if (threadIdx.x == 0) {
		atomicAdd(result, blockSum);
	}
}

__global__ void NormUnrolled(const float* vec, size_t size, float* result) {
}

__global__ void NormDiff(const float* vec1, const float* vec2, size_t size, float* result) {
	extern __shared__ float blockParSums[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We calculate the squared value
	float data = vec1[idx] - vec2[idx];
	data = data * data;

	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	float warpSum = WarpReduce(data);
	if (warpLane == 0) blockParSums[warpIndex] = warpSum;

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 

	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;
	// Facciamo lavorare solo il primo warp
	float blockSum = warpIndex == 0 ? WarpReduce(warpParSum) : 0.0f;

	if (threadIdx.x == 0) {
		atomicAdd(result, blockSum);
	}
}
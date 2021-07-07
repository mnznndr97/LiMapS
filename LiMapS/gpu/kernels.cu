#include "kernels.cuh"





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



__device__ void Norm(const float* vec, size_t size, float* result) {
	dim3 blockSize(256);
	SquareSumKrnl << <(size + blockSize.x - 1) / blockSize.x, blockSize.x, blockSize.x / 32 >> > (vec, size, result);
	CUDA_CHECKD(cudaDeviceSynchronize());
	assert(*result >= 0.0f);

	*result = sqrtf(*result);
}

__global__ void NormUnrolled(const float* vec, size_t size, float* result) {
}

__device__ void NormDiff(const float* vec1, const float* vec2, size_t size, float* result) {
	dim3 blockSize(256);
	NormDiffKrn << <(size + blockSize.x - 1) / blockSize.x, blockSize.x, blockSize.x / warpSize >> > (vec1, vec2, size, result);
	CUDA_CHECKD(cudaDeviceSynchronize());

	assert(*result >= 0.0f);
	*result = sqrtf(*result);
}

__global__ void NormDiffKrn(const float* vec1, const float* vec2, size_t size, float* result) {
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
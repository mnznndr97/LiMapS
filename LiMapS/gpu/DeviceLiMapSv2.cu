#include "DeviceLiMapSv2.cuh"

#include "cuda_shared.h"
#include <cooperative_groups.h>
#include <cuda/std/functional>
#include "cublas_shared.h"

#include "kernels.cuh"
#include "threshold_kernels.cuh"



__device__ float* _solutionD;
__device__ float* _signalD;
__device__ float* _dictionaryD;
__device__ float* _dictionaryInverseD;
__device__ float* _alphaD;
__device__ float* _alphaOldD;

__device__ float* _beta;
__device__ float* _intermD;

__device__ float _signalSquareSum;
__device__ float _alphaDiffSquareSum;

template<int unrollFactor>
__global__ void FillZero(float* vector, size_t size) {
	cg::grid_group grid = cg::this_grid();
	if (grid.thread_rank() >= size) {
		// Our thread is out of range
		return;
	}

#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = grid.thread_rank() + i * blockDim.x;
		vector[vOffset] = 0.0f;
	}
}


__global__ void GetAlpha(size_t dictionaryWords, size_t signalSize) {
	cg::grid_group grid = cg::this_grid();
	if (grid.thread_rank() >= dictionaryWords) {
		// Our thread is out of range
		return;
	}

	float sum = 0.0f;
	for (size_t i = 0; i < signalSize; i++)
	{
		sum += _dictionaryInverseD[grid.thread_rank() * signalSize + i] * _signalD[i];
		//sum = fmaf(_dictionaryInverseD[grid.thread_rank() * signalSize + i], _signalD[i], sum);
	}

	// AlphaOld and Alpha should be aligned at the first iteration, so let's write them directly here
	// This avoids us a vector copy later
	_alphaD[grid.thread_rank()] = sum;
	_alphaOldD[grid.thread_rank()] = sum;
}

template<int unrollFactor>
__global__ void GetAlpha2(size_t dictionaryWords, size_t signalSize) {
	extern __shared__ float blockParSums[];

	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	size_t idy = blockDim.y * blockIdx.y + threadIdx.y;

	if (idy >= dictionaryWords) return;

	float data = 0.0f;
#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float dicInverse = vOffset < signalSize ? _dictionaryInverseD[idy * signalSize + vOffset] : 0.0f;
		float signal = vOffset < signalSize ? _signalD[vOffset] : 0.0f;

		data += (dicInverse * signal);
	}


	int warpLane = threadIdx.x % warpSize;
	int warpIndex = threadIdx.x / warpSize;

	// We sum the data at warp level and we store the value of the first thread (at warp level) in the 
	// shared memory
	float warpSum = WarpReduce(data);
	if (warpLane == 0) {
		// NB: if we are in the "out-of-bounds" region of the data, sum shound be zero 
		// This is necessary to have a legitimate value in the shared mem that later will 
		// be used in another reduce pass
		blockParSums[warpIndex] = warpSum;
	}

	if (idx >= signalSize) return;

	__syncthreads();

	// After the sync point, we can use the first threads (up to shared mem size), to load and sum toghether the 
	// values at "block level", 

	float warpParSum = (threadIdx.x < (blockDim.x / warpSize)) ? blockParSums[threadIdx.x] : 0.0f;

	// Facciamo lavorare solo il primo warp
	float blockSum = warpIndex == 0 ? WarpReduce(warpParSum) : 0.0f;

	// AlphaOld and Alpha should be aligned at the first iteration, so let's write them directly here
	// This avoids us a vector copy later
	if (threadIdx.x == 0) {
		atomicAdd(&_alphaD[idy], blockSum);
		atomicAdd(&_alphaOldD[idy], blockSum);
	}
}

__global__ void CalculateBetaStep(float lambda, size_t dictionaryWords, size_t signalSize) {
	cg::grid_group grid = cg::this_grid();

	unsigned long long index = grid.thread_rank();
	if (index >= dictionaryWords) {
		// Our thread is out of range
		return;
	}

	_beta[index] = GetBeta(lambda, _alphaD[index]);
}

__global__ void CalculateIntermStep(size_t dictionaryWords, size_t signalSize) {
	cg::grid_group grid = cg::this_grid();

	unsigned long long idx = grid.thread_rank();
	if (idx >= signalSize) {
		// Our thread is out of range
		return;
	}

	float sum = 0.0f;
	for (size_t i = 0; i < dictionaryWords; i++)
	{
		sum += _dictionaryD[idx * dictionaryWords + i] * _beta[i];
		//sum = fmaf(_dictionaryD[idx * dictionaryWords + i], _beta[i], sum);
	}
	_intermD[idx] = sum - _signalD[idx];
}

__global__ void CalculateNewAlphaStep(size_t dictionaryWords, size_t signalSize) {
	cg::grid_group grid = cg::this_grid();

	unsigned long long idx = grid.thread_rank();
	if (idx >= dictionaryWords) {
		// Our thread is out of range
		return;
	}

	float sum = 0.0f;
	for (size_t i = 0; i < signalSize; i++)
	{
		sum += _dictionaryInverseD[idx * signalSize + i] * _intermD[i];
		//sum = fmaf(_dictionaryInverseD[idx * signalSize + i], _intermD[i], sum);
	}
	float newAlpha = _beta[idx] - sum;
	_alphaD[idx] = fabs(newAlpha) >= 1e-4f ? newAlpha : 0.0f;
}


__global__ void LiMapS(size_t dictionaryWords, size_t signalSize) {
	// Handle to thread block group
	cg::grid_group grid = cg::this_grid();


	// 1) The first step of the LiMapS algorithm is to calculate the starting lamba coefficient. In order to do so, we need to calculate
	// the signal norm. So we enqueue on the default stream the SquareSum operation and then we wait for it.
	// The norm is foundamental for the next steps so there is nothing that we can do to avoid the sync time waste
	_signalSquareSum = 0.0f;
	dim3 blocks(256);
	SquareSumKrnlUnroll<8> << <GetGridSize(blocks, signalSize, 8), blocks, blocks.x / warpSize >> > (_signalD, signalSize, &_signalSquareSum);
	CUDA_CHECKD(cudaDeviceSynchronize());

	assert(_signalSquareSum >= 0.0f);

	float t = sqrtf(_signalSquareSum);
	float lambda = 1.0f / t;

	_beta = new float[dictionaryWords];
	_intermD = new float[signalSize];

	blocks.x = 32;
	// 2) The second step of the algorithm is to prepare the starting alpha vector so also here we 
	// Launch the kernel calculation and we synchronize the device

	// Is it  necessary??
	//FillZero<1> << <gridSize, blocks >> > (_alphaD, dictionaryWords);
	//FillZero<1> << <gridSize, blocks >> > (_alphaOldD, dictionaryWords);

	dim3 gridSize = GetGridSize(blocks, signalSize, 8);
	gridSize.y = dictionaryWords;
	int sharedMemSize = blocks.x / warpSize;
	GetAlpha2<8> << <gridSize, blocks, sharedMemSize >> > (dictionaryWords, signalSize);
	CUDA_CHECKD(cudaPeekAtLastError());
	CUDA_CHECKD(cudaDeviceSynchronize());

	int i = 0;
	for (i = 0; i < 1000; i++)
	{
		// We set the alphaOld as the current alpha. We can do this by just swapping the pointer, avoiding 
		// useless data transfer
		cuda::std::swap(_alphaD, _alphaOldD);

		// From here, we split our computation next alpha computation in different step. This is necessary since some calculation
		// depend on data that should accessed after a global sync point (ex, after calculating the intermediate (dic * beta - sig) vector
		// Since global sync CANNOT be achieved (at least in old devices that not support grid_group::sync() method), we can do better:
		// we just queue our splitted work on the default stream, and then we just sync with the device at the end from this kenel.
		// In this way, the work is executed with all data dependencies respected

		// 3.1) We need to compute the beta vector for this iterarion
		blocks.x = 128;
		blocks.y = 1;
		CalculateBetaStep << <GetGridSize(blocks, dictionaryWords), blocks >> > (lambda, dictionaryWords, signalSize);

		// 3.2) We need to compute the intermediate (dic * beta - sig) vector
		CalculateIntermStep << <GetGridSize(blocks, signalSize), blocks >> > (dictionaryWords, signalSize);

		// 3.3) We compute the new alpha with the thresholding at the end
		CalculateNewAlphaStep << <GetGridSize(blocks, dictionaryWords), blocks >> > (dictionaryWords, signalSize);

		lambda = 1.01f * lambda;

		// 3.4) We see how much alpha is changed
		_alphaDiffSquareSum = 0.0f;
		SquareDiffSumKrnlUnroll<8> << <GetGridSize(blocks, dictionaryWords, 8), blocks >> > (_alphaD, _alphaOldD, dictionaryWords, &_alphaDiffSquareSum);
		CUDA_CHECKD(cudaDeviceSynchronize());

		float norm = sqrtf(_alphaDiffSquareSum);
		if (norm < 1e-5f) {
			break;
		}
	}

	printf("kernel iterations: %d\r\n", i);
	delete[] _beta;
	delete[] _intermD;
}

DeviceLiMapSv2::DeviceLiMapSv2(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize)
	: BaseLiMapS(solution, signal, D, DINV, dictionaryWords, signalSize)
{
	_alphaH.resize(_dictionaryWords);

	// We create the cuda pointers here and then we copy the pointers values to the device symbols. In this way
	// memory disposal should be automatically handled by the class
	_solutionPtr = make_cuda<float>(dictionaryWords);
	_signalPtr = make_cuda<float>(signalSize);
	_dictionaryPtr = make_cuda<float>(dictionaryWords * signalSize);
	_dictionaryInversePtr = make_cuda<float>(dictionaryWords * signalSize);
	_alphaPtr = make_cuda<float>(dictionaryWords);
	_alphaOldPtr = make_cuda<float>(dictionaryWords);

	float* dummyPtr = _solutionPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_solutionD, &dummyPtr, sizeof(void*)));

	dummyPtr = _signalPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_signalD, &dummyPtr, sizeof(void*)));

	dummyPtr = _dictionaryPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_dictionaryD, &dummyPtr, sizeof(void*)));

	dummyPtr = _dictionaryInversePtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_dictionaryInverseD, &dummyPtr, sizeof(void*)));

	dummyPtr = _alphaPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_alphaD, &dummyPtr, sizeof(void*)));

	dummyPtr = _alphaOldPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_alphaOldD, &dummyPtr, sizeof(void*)));
}


void DeviceLiMapSv2::Execute(int iterations)
{
	CUDA_CHECK(cudaMemcpyAsync(_signalPtr.get(), _signalHost, sizeof(float) * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(_dictionaryInversePtr.get(), _dictionaryInverseHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(_dictionaryPtr.get(), _dictionaryHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));

	LiMapS << < 1, 1 >> > (_dictionaryWords, _signalSize);

	CUDA_CHECK(cudaMemcpyAsync(_alphaH.data(), _alphaPtr.get(), sizeof(float) * _dictionaryWords, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());
}

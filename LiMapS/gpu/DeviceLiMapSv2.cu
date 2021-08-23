#include "DeviceLiMapSv2.cuh"

#include "cuda_shared.h"
#include <cooperative_groups.h>
#include <cuda/std/functional>
#include "cublas_shared.h"

#include "kernels/misc.cuh"
#include "kernels/square_sum.cuh"
#include "kernels/matrix2vector.cuh"

static __device__ float* _solutionD;
static __device__ float* _signalD;
static __device__ float* _dictionaryD;
static __device__ float* _dictionaryInverseD;
static __device__ float* _alphaD;
static __device__ float* _alphaNewD;

static __device__ float* _beta;
static __device__ float* _intermD;

static __device__ float _signalSquareSum;
static __device__ float _alphaDiffSquareSum;


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
	_alphaNewD[grid.thread_rank()] = sum;
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
	_alphaNewD[idx] = fabs(newAlpha) >= 1e-4f ? newAlpha : 0.0f;
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

	blocks.x = 128;
	// 2) The second step of the algorithm is to prepare the starting alpha vector so also here we 
	// Launch the kernel calculation and we synchronize the device

	GetAlpha << <GetGridSize(blocks, dictionaryWords), blocks >> > (dictionaryWords, signalSize);
	CUDA_CHECKD(cudaDeviceSynchronize());

	int i = 0;
	for (i = 0; i < 1000; i++)
	{
		// We set the alphaOld as the current alpha. We can do this by just swapping the pointer, avoiding 
		// useless data transfer
		cuda::std::swap(_alphaD, _alphaNewD);

		// From here, we split our computation next alpha computation in different step. This is necessary since some calculation
		// depend on data that should accessed after a global sync point (ex, after calculating the intermediate (dic * beta - sig) vector
		// Since global sync CANNOT be achieved (at least in old devices that not support grid_group::sync() method), we can do better:
		// we just queue our splitted work on the default stream, and then we just sync with the device at the end from this kenel.
		// In this way, the work is executed with all data dependencies respected

		// 3.1) We need to compute the beta vector for this iterarion
		blocks.x = 128;
		CalculateBeta<1> << <GetGridSize(blocks, dictionaryWords), blocks >> > (_alphaD, _beta, lambda, dictionaryWords);

		// 3.2) We need to compute the intermediate (dic * beta - sig) vector
		blocks.x = 64;
		CalculateIntermStep << <GetGridSize(blocks, signalSize), blocks >> > (dictionaryWords, signalSize);

		// 3.3) We compute the new alpha with the thresholding at the end
		blocks.x = 64;
		CalculateNewAlphaStep << <GetGridSize(blocks, dictionaryWords), blocks >> > (dictionaryWords, signalSize);

		lambda = 1.01f * lambda;

		// 3.4) We see how much alpha is changed
		_alphaDiffSquareSum = 0.0f;
		SquareDiffSumKrnlUnroll<8> << <GetGridSize(blocks, dictionaryWords, 8), blocks >> > (_alphaNewD, _alphaD, dictionaryWords, &_alphaDiffSquareSum);
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
	CUDA_CHECK(cudaMemcpyToSymbol(_alphaNewD, &dummyPtr, sizeof(void*)));
}


void DeviceLiMapSv2::Execute(int iterations)
{
	// We lanuch the memory copies asyncronously here and then we wait on the sync point and the end of the function
	// In this way we first enqueue all the work on the NULL stream and then we waiting, minimizing the "wasted" time in CPU-GPU
	// command execution
	CUDA_CHECK(cudaMemcpyAsync(_signalPtr.get(), _signalHost, sizeof(float) * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(_dictionaryInversePtr.get(), _dictionaryInverseHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(_dictionaryPtr.get(), _dictionaryHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));

	// LiMapS kernel will dynamically launch its own kernels. So only one thread is necessary
	// By doing this, we can erase the CPU-GPU communication time for launching kernels
	LiMapS << < 1, 1 >> > (_dictionaryWords, _signalSize);

	CUDA_CHECK(cudaMemcpyAsync(_alphaH.data(), _alphaPtr.get(), sizeof(float) * _dictionaryWords, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());
}

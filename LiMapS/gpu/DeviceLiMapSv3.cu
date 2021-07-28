#include "DeviceLiMapSv3.cuh"

#include <iostream>
#include "cuda_shared.h"
#include <cooperative_groups.h>
#include <cuda/std/functional>
#include "cublas_shared.h"

#include "kernels.cuh"

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

template<int unrollFactor>
__global__ void FillInterm(float* vector, size_t size) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		if (vOffset < size) vector[vOffset] = -_signalD[vOffset];
	}
}

template<int unrollFactor>
__global__ void GetAlphaImprv(size_t dictionaryWords, size_t signalSize) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy >= dictionaryWords) return;

	float data = 0.0f;
#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float dicInverse = vOffset < signalSize ? _dictionaryInverseD[idy * signalSize + vOffset] : 0.0f;
		float signal = vOffset < signalSize ? _signalD[vOffset] : 0.0f;

		//data += (dicInverse * signal);
		data = fmaf(dicInverse, signal, data);
	}

	KernelReduce<float*, float*>(data, signalSize, [](float* ptr1, float* ptr2, float sum) {
		atomicAdd(ptr1, sum);
		atomicAdd(ptr2, sum);
		}, &_alphaD[idy], &_alphaNewD[idy]);
}

__global__ void LiMapS2(size_t dictionaryWords, size_t signalSize) {
	// 1) The first step of the LiMapS algorithm is to calculate the starting lamba coefficient. In order to do so, we need to calculate
	// the signal norm. So we enqueue on the default stream the SquareSum operation and then we wait for it.
	// The norm is foundamental for the next steps so there is nothing that we can do to avoid the sync time waste
	_signalSquareSum = 0.0f;
	dim3 blocks(128);
	dim3 red8DicGridSize = GetGridSize(blocks, dictionaryWords, 8);
	dim3 red8SignalGridSize = GetGridSize(blocks, signalSize, 8);

	SquareSumKrnlUnroll<8> << <red8SignalGridSize, blocks, blocks.x / warpSize >> > (_signalD, signalSize, &_signalSquareSum);
	CUDA_CHECKD(cudaDeviceSynchronize());

	assert(_signalSquareSum >= 0.0f);

	float t = sqrtf(_signalSquareSum);
	float lambda = 1.0f / t;

	_beta = new float[dictionaryWords];
	_intermD = new float[signalSize];


	// 2) The second step of the algorithm is to prepare the starting alpha vector so also here we 
	// Launch the kernel calculation and we synchronize the device

	// Is it  necessary??
	//FillZero<1> << <gridSize, blocks >> > (_alphaD, dictionaryWords);
	//FillZero<1> << <gridSize, blocks >> > (_alphaOldD, dictionaryWords);


	dim3 gridSize = red8SignalGridSize;
	gridSize.y = dictionaryWords;
	int sharedMemSize = blocks.x / warpSize;
	GetAlphaImprv<8> << <gridSize, blocks, sharedMemSize >> > (dictionaryWords, signalSize);
	CUDA_CHECKD(cudaPeekAtLastError());
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
		CalculateBeta<8> << <red8DicGridSize, blocks, 0 >> > (_alphaD, _beta, lambda, dictionaryWords);

		// 3.2) We need to compute the intermediate (dic * beta - sig) vector
		CopyTo<8> << <red8SignalGridSize, blocks, 0 >> > (_signalD, signalSize, _intermD, true);

		gridSize = red8DicGridSize;
		//gridSize.x = 80;
		gridSize.y = signalSize;
		int sharedMemSize = blocks.x / warpSize;
		Matrix2Vector<8, false> << <gridSize, blocks, sharedMemSize >> > (_dictionaryD, _beta, _intermD, dictionaryWords, signalSize);
		CUDA_CHECKD(cudaPeekAtLastError());

		// 3.3) We compute the new alpha with the thresholding at the end
		CopyTo<8> << <red8DicGridSize, blocks, 0 >> > (_beta, dictionaryWords, _alphaNewD, false);

		blocks.y = 1;
		gridSize = red8SignalGridSize;
		//gridSize.x = 80;
		gridSize.y = (dictionaryWords + 1) / 2;
		gridSize.y = dictionaryWords;
		sharedMemSize = blocks.x / warpSize;
		Matrix2Vector<8, true> << <gridSize, blocks, sharedMemSize >> > (_dictionaryInverseD, _intermD, _alphaNewD, signalSize, dictionaryWords);
		CUDA_CHECKD(cudaPeekAtLastError());

		// NB: Benchmarks says that 128 threads per block should result in the best occupancy for the
		// threshold kernel
		blocks.y = 1;
		ThresholdVector<8> << <red8DicGridSize, blocks >> > (_alphaNewD, dictionaryWords);

		lambda = 1.01f * lambda;

		// 3.4) We see how much alpha is changed
		_alphaDiffSquareSum = 0.0f;
		SquareDiffSumKrnlUnroll<8> << <red8DicGridSize, blocks, sharedMemSize >> > (_alphaNewD, _alphaD, dictionaryWords, &_alphaDiffSquareSum);
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

DeviceLiMapSv3::DeviceLiMapSv3(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize)
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


void DeviceLiMapSv3::Execute(int iterations)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// We lanuch the memory copies asyncronously here and then we wait on the sync point and the end of the function
	// In this way we first enqueue all the work on the NULL stream and then we waiting, minimizing the "wasted" time in CPU-GPU
	// command execution
	CUDA_CHECK(cudaMemcpyAsync(_signalPtr.get(), _signalHost, sizeof(float) * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(_dictionaryInversePtr.get(), _dictionaryInverseHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpyAsync(_dictionaryPtr.get(), _dictionaryHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));

	// LiMapS kernel will dynamically launch its own kernels. So only one thread is necessary
	// By doing this, we can erase the CPU-GPU communication time for launching kernels
	cudaEventRecord(start);
	LiMapS2 << < 1, 1 >> > (_dictionaryWords, _signalSize);
	cudaEventRecord(stop);

	CUDA_CHECK(cudaMemcpyAsync(_alphaH.data(), _alphaPtr.get(), sizeof(float) * _dictionaryWords, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());

	// Let's just also measure the kernel exec time to see the mem copies/sync overhead
	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Event elapsed: " << ms << " ms" << std::endl;
}

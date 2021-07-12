#include "DeviceLiMapSTex.cuh"

#include <iostream>
#include "cuda_shared.h"
#include <cooperative_groups.h>
#include <cuda/std/functional>
#include "cublas_shared.h"

#include "kernels.cuh"
#include "threshold_kernels.cuh"



static __device__ float* _solutionD;
static __device__ float* _signalD;
static __device__ float* _alphaD;
static __device__ float* _alphaOldD;
static __device__ float* _beta;
static __device__ float* _intermD;

static __device__ float _signalSquareSum;
static __device__ float _alphaDiffSquareSum;

template<int unrollFactor>
static __global__ void FillInterm(float* vector, size_t size) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		if (vOffset < size) vector[vOffset] = -_signalD[vOffset];
	}
}

template<int unrollFactor>
static __global__ void FillAlpha(float* vector, size_t size) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		if (vOffset < size) vector[vOffset] = _beta[vOffset];
	}
}

template<int unrollFactor>
static __global__ void ThresholdAlpha(float* vector, size_t size) {
	int idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;

#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		if (vOffset < size) {
			if (fabs(_alphaD[vOffset]) < 1e-4f)
				_alphaD[vOffset] = 0.0f;
		}

	}
}

template<int unrollFactor>
static __global__ void GetAlpha2(cudaTextureObject_t dictionaryInverseTexture, size_t dictionaryWords, size_t signalSize) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy >= dictionaryWords) return;

	float data = 0.0f;
#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float dicInverse = vOffset < signalSize ? tex2D<float>(dictionaryInverseTexture, vOffset, idy) : 0.0f;
		float signal = vOffset < signalSize ? _signalD[vOffset] : 0.0f;

		data += (dicInverse * signal);
	}

	KernelReduce<size_t>(data, signalSize, [](size_t index, float sum) {
		atomicAdd(&_alphaD[index], sum);
		atomicAdd(&_alphaOldD[index], sum);
		}, idy);
}

static __global__ void CalculateBetaStep2(float lambda, size_t dictionaryWords, size_t signalSize) {
	cg::grid_group grid = cg::this_grid();

	unsigned long long index = grid.thread_rank();
	if (index >= dictionaryWords) {
		// Our thread is out of range
		return;
	}

	_beta[index] = GetBeta(lambda, _alphaD[index]);
}


template<int unrollFactor>
static __global__ void CalculateIntermStep2(cudaTextureObject_t dictionaryTexture, size_t dictionaryWords, size_t signalSize) {

	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy >= signalSize) return;

	float data = 0.0f;
#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float dic = vOffset < dictionaryWords ? tex2D<float>(dictionaryTexture, vOffset, idy) : 0.0f;
		float beta = vOffset < dictionaryWords ? _beta[vOffset] : 0.0f;

		data += (dic * beta);
		//data = fma(dic, beta, data);
	}

	KernelReduce<size_t>(data, dictionaryWords, [](size_t index, float sum) {
		atomicAdd(&_intermD[index], sum);
		}, idy);
}


template<int unrollFactor>
static __global__ void CalculateNewAlphaStep2(cudaTextureObject_t dictionaryInverseTexture, size_t dictionaryWords, size_t signalSize) {
	size_t idx = blockIdx.x * (blockDim.x * unrollFactor) + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy >= dictionaryWords) return;

	float data = 0.0f;
#pragma unroll
	for (size_t i = 0; i < unrollFactor; i++)
	{
		size_t vOffset = idx + i * blockDim.x;
		float dicInv = vOffset < signalSize ? tex2D<float>(dictionaryInverseTexture, vOffset, idy) : 0.0f;
		float interm = vOffset < signalSize ? _intermD[vOffset] : 0.0f;

		data += (dicInv * interm);
	}

	KernelReduce<size_t>(data, signalSize, [](size_t index, float sum) {
		atomicAdd(&_alphaD[index], -sum);
		}, idy);
}


static __global__ void LiMapS(cudaTextureObject_t dictionaryTexture, cudaTextureObject_t dictionaryInverseTexture, size_t dictionaryWords, size_t signalSize) {

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
	GetAlpha2<8> << <gridSize, blocks, sharedMemSize >> > (dictionaryInverseTexture, dictionaryWords, signalSize);
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
		CalculateBetaStep2 << <GetGridSize(blocks, dictionaryWords), blocks, 0 >> > (lambda, dictionaryWords, signalSize);

		// 3.2) We need to compute the intermediate (dic * beta - sig) vector
		blocks.x = 64;
		FillInterm<8> << <GetGridSize(blocks, signalSize, 8), blocks, 0 >> > (_intermD, signalSize);

		blocks.x = 64;
		gridSize = GetGridSize(blocks, dictionaryWords, 8);
		gridSize.y = signalSize;
		int sharedMemSize = blocks.x / warpSize;
		CalculateIntermStep2<8> << <gridSize, blocks, sharedMemSize >> > (dictionaryTexture, dictionaryWords, signalSize);
		CUDA_CHECKD(cudaPeekAtLastError());

		// 3.3) We compute the new alpha with the thresholding at the end

		blocks.x = 64;
		FillAlpha<8> << <GetGridSize(blocks, dictionaryWords, 8), blocks, 0 >> > (_alphaD, dictionaryWords);

		blocks.x = 128;
		blocks.y = 1;
		gridSize = GetGridSize(blocks, signalSize, 8);
		gridSize.y = (dictionaryWords + 7) / 8;
		gridSize.y = dictionaryWords;
		sharedMemSize = blocks.x / warpSize;
		CalculateNewAlphaStep2<8> << <gridSize, blocks, sharedMemSize >> > (dictionaryInverseTexture, dictionaryWords, signalSize);
		CUDA_CHECKD(cudaPeekAtLastError());

		blocks.x = 128;
		blocks.y = 1;
		ThresholdAlpha<8> << <GetGridSize(blocks, dictionaryWords, 8), blocks >> > (_alphaD, dictionaryWords);

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

DeviceLiMapSTex::DeviceLiMapSTex(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize)
	: BaseLiMapS(solution, signal, D, DINV, dictionaryWords, signalSize)
{
	_alphaH.resize(_dictionaryWords);

	// We create the cuda pointers here and then we copy the pointers values to the device symbols. In this way
	// memory disposal should be automatically handled by the class
	_solutionPtr = make_cuda<float>(dictionaryWords);
	_signalPtr = make_cuda<float>(signalSize);

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	CUDA_CHECK(cudaMallocArray(&_dictionaryArray, &channelDesc, dictionaryWords, signalSize));
	CUDA_CHECK(cudaMallocArray(&_dictionaryInverseArray, &channelDesc, signalSize, dictionaryWords));

	_alphaPtr = make_cuda<float>(dictionaryWords);
	_alphaOldPtr = make_cuda<float>(dictionaryWords);

	float* dummyPtr = _solutionPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_solutionD, &dummyPtr, sizeof(void*)));

	dummyPtr = _signalPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_signalD, &dummyPtr, sizeof(void*)));

	dummyPtr = _alphaPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_alphaD, &dummyPtr, sizeof(void*)));

	dummyPtr = _alphaOldPtr.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_alphaOldD, &dummyPtr, sizeof(void*)));
}

DeviceLiMapSTex::~DeviceLiMapSTex()
{
	CUDA_CHECK(cudaFreeArray(_dictionaryArray));
	CUDA_CHECK(cudaFreeArray(_dictionaryInverseArray));
}


void DeviceLiMapSTex::Execute(int iterations)
{
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	CUDA_CHECK(cudaMemcpyAsync(_signalPtr.get(), _signalHost, sizeof(float) * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy2DToArrayAsync(_dictionaryArray, 0, 0, _dictionaryHost, _dictionaryWords * sizeof(float), _dictionaryWords * sizeof(float), _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy2DToArrayAsync(_dictionaryInverseArray, 0, 0, _dictionaryInverseHost, _signalSize * sizeof(float), _signalSize * sizeof(float), _dictionaryWords, cudaMemcpyHostToDevice));

	// Specify texture
	struct cudaResourceDesc dictionaryResDesc;
	memset(&dictionaryResDesc, 0, sizeof(dictionaryResDesc));
	dictionaryResDesc.resType = cudaResourceTypeArray;
	dictionaryResDesc.res.array.array = _dictionaryArray;

	struct cudaResourceDesc dictionaryInvResDesc;
	memset(&dictionaryInvResDesc, 0, sizeof(dictionaryInvResDesc));
	dictionaryInvResDesc.resType = cudaResourceTypeArray;
	dictionaryInvResDesc.res.array.array = _dictionaryInverseArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	// When addressing our texture, we should ALWAYS use the correct coordinates
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	// No interpolation either
	texDesc.filterMode = cudaFilterModePoint;
	// We don't want any normalization in input/output 
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	cudaTextureObject_t dictionaryTexture = 0;
	cudaTextureObject_t dictionaryInverseTexture = 0;
	cudaCreateTextureObject(&dictionaryTexture, &dictionaryResDesc, &texDesc, NULL);
	cudaCreateTextureObject(&dictionaryInverseTexture, &dictionaryInvResDesc, &texDesc, NULL);


	cudaEventRecord(start);
	LiMapS << < 1, 1 >> > (dictionaryTexture, dictionaryInverseTexture, _dictionaryWords, _signalSize);
	cudaEventRecord(stop);

	CUDA_CHECK(cudaMemcpyAsync(_alphaH.data(), _alphaPtr.get(), sizeof(float) * _dictionaryWords, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());

	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Event elapsed: " << ms << " ms" << std::endl;

	cudaDestroyTextureObject(dictionaryTexture);
	cudaDestroyTextureObject(dictionaryInverseTexture);
}

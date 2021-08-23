#include "DeviceLiMapSv4.cuh"

#include <iostream>
#include "cuda_shared.h"
#include <cooperative_groups.h>
#include <cuda/std/functional>
#include "cublas_shared.h"

#include "kernels/misc.cuh"
#include "kernels/square_sum.cuh"
#include "kernels/matrix2vector.cuh"
#include "kernels/transpose.cuh"

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

__global__ void LiMapS4(size_t dictionaryWords, size_t signalSize) {
	// 1) The first step of the LiMapS algorithm is to calculate the starting lamba coefficient. In order to do so, we need to calculate
	// the signal norm. So we enqueue on the default stream the SquareSum operation and then we wait for it.
	// The norm is foundamental for the next steps so there is nothing that we can do to avoid the sync time waste
	_signalSquareSum = 0.0f;
	dim3 blocks(128);
	dim3 red8DicGridSize = GetGridSize(blocks, dictionaryWords, 8);
	dim3 red8SignalGridSize = GetGridSize(blocks, signalSize, 8);
	dim3 dicGridSize = GetGridSize(blocks, dictionaryWords);
	dim3 signalGridSize = GetGridSize(blocks, signalSize);
	int sharedMemSize = blocks.x / warpSize;

	SquareSumKrnlUnroll<8> << <red8SignalGridSize, blocks, sharedMemSize >> > (_signalD, signalSize, &_signalSquareSum);
	CUDA_CHECKD(cudaDeviceSynchronize());

	assert(_signalSquareSum >= 0.0f);
	float lambda = 1.0f / sqrtf(_signalSquareSum);

	_beta = new float[dictionaryWords];
	_intermD = new float[signalSize];

	// 2) The second step of the algorithm is to prepare the starting alpha vector so also here we 
	// Launch the kernel calculation and we synchronize the device

	Matrix2Vector2 << <dicGridSize, blocks >> > (_dictionaryInverseD, _signalD, _alphaD, signalSize, dictionaryWords, [](float* dest2, float* dest, size_t index, float sum) {
		dest[index] = sum;
		dest2[index] = sum;
		}, _alphaNewD);
	CUDA_CHECKD(cudaDeviceSynchronize());

	dim3 gridSize = signalGridSize;
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
		Matrix2Vector2 << <signalGridSize, blocks >> > (_dictionaryD, _beta, _intermD, dictionaryWords, signalSize, [](float* signal, float* dest, size_t index, float sum) {
			dest[index] = sum - signal[index];
			}, _signalD);

		// 3.3) We compute the new alpha with the thresholding at the end
		Matrix2Vector2 << <dicGridSize, blocks >> > (_dictionaryInverseD, _intermD, _alphaNewD, signalSize, dictionaryWords, [](float* beta, float* dest, size_t index, float sum) {
			float data = beta[index] - sum;
			dest[index] = fabs(data) < 1e-4f ? 0.0f : data;
			}, _beta);

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

DeviceLiMapSv4::DeviceLiMapSv4(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize)
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


void DeviceLiMapSv4::Execute(int iterations)
{
	/* First prparation step: we need to transpose the two dictionaries since our Matrxi2Vector2 needs a column major ordering */
	cuda_ptr<float> tempDic = make_cuda<float>(_dictionaryWords * _signalSize);

	CUDA_CHECK(cudaMemcpyAsync(tempDic.get(), _dictionaryHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));
	// Let's calculate the transpose necessary grid dimension and blocks
	dim3 txGrid((_dictionaryWords + TILE_DIM - 1) / TILE_DIM, (_signalSize + TILE_DIM - 1) / TILE_DIM);
	dim3 txThreads(TILE_DIM, BLOCK_ROWS);
	Transpose << < txGrid, txThreads >> > (tempDic.get(), _dictionaryPtr.get(), _dictionaryWords, _signalSize);

	CUDA_CHECK(cudaMemcpyAsync(tempDic.get(), _dictionaryInverseHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));
	std::swap(txGrid.x, txGrid.y);
	Transpose << < txGrid, txThreads >> > (tempDic.get(), _dictionaryInversePtr.get(), _signalSize, _dictionaryWords);

	// We must syncronize to release our pointer since we are launching the kernel and the copies asynchronously
	CUDA_CHECK(cudaDeviceSynchronize());
	tempDic.release();

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// We launch the memory copies asyncronously here and then we wait on the sync point and the end of the function
	// In this way we first enqueue all the work on the NULL stream and then we wait, minimizing the "wasted" time in CPU-GPU
	// command execution
	CUDA_CHECK(cudaMemcpyAsync(_signalPtr.get(), _signalHost, sizeof(float) * _signalSize, cudaMemcpyHostToDevice));

	// LiMapS kernel will dynamically launch its own kernels. So only one thread is necessary
	// By doing this, we can erase the CPU-GPU communication time for launching kernels
	cudaEventRecord(start);
	LiMapS4 << < 1, 1 >> > (_dictionaryWords, _signalSize);
	cudaEventRecord(stop);

	CUDA_CHECK(cudaMemcpyAsync(_alphaH.data(), _alphaPtr.get(), sizeof(float) * _dictionaryWords, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaDeviceSynchronize());

	// Let's just also measure the kernel exec time to see the mem copies/sync overhead
	float ms;
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "Event elapsed: " << ms << " ms" << std::endl;
}

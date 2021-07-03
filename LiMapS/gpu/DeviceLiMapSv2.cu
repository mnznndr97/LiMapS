#include "DeviceLiMapSv2.cuh"

#include "cublas_shared.h"
#include "beta_kernels.cuh"
#include "threshold_kernels.cuh"


__device__ float* _solutionD;
__device__ float* _signalD;
__device__ float* _dictionaryD;
__device__ float* _dictionaryInverseD;
__device__ float* _alphaD;
__device__ float* _alphaOldD;

__device__ float _signalNorm;

__global__ void Norm(const float* vec, size_t size, float* result) {
}

__global__ void LiMapS(size_t dictionaryWords, size_t signalSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx == 0) {
		dim3 blockSize(32);
		dim3 gridSize((signalSize + blockSize.x - 1) / blockSize.x);
		Norm << <gridSize, blockSize >> > (_signalD, signalSize, &_signalNorm);
	}
	CUDA_CHECKD(cudaDeviceSynchronize());

	int a = (int)_signalNorm;
}

DeviceLiMapSv2::DeviceLiMapSv2(std::vector<float>& solution, std::vector<float>& signal, std::vector<float>& D, std::vector<float>& DINV)
	:_signalSize(signal.size()), _dictionaryWords(solution.size()),
	// To avoid C++ vector copies, let's just store the vector references for our input data. This may be dangerous since the class MUST have the same (or shorted)
	// scope of our data, but for our purposes should be ok
	_hostSolution(solution), _hostSignal(signal), _hostDictionary(D), _hostDictionaryInverse(DINV)
{
	_solution = make_cuda<float>(solution.size());
	_signal = make_cuda<float>(signal.size());
	_dictionary = make_cuda<float>(D.size());
	_dictionaryInverse = make_cuda<float>(DINV.size());

	_alpha = make_cuda<float>(solution.size());
	_alphaOld = make_cuda<float>(solution.size());

	// We copy the 

	float* dummyPtr = _solution.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_solutionD, &dummyPtr, sizeof(void*)));

	dummyPtr = _signal.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_signalD, &dummyPtr, sizeof(void*)));

	dummyPtr = _dictionary.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_dictionaryD, &dummyPtr, sizeof(void*)));

	dummyPtr = _dictionaryInverse.get();
	CUDA_CHECK(cudaMemcpyToSymbol(_dictionaryInverseD, &dummyPtr, sizeof(void*)));
}

DeviceLiMapSv2::~DeviceLiMapSv2()
{
	
}

void DeviceLiMapSv2::Execute(int iterations)
{
	dim3 blocks(32);
	dim3 gridSize((_dictionaryWords + blocks.x - 1) / blocks.x);

	LiMapS << < gridSize, blocks >> > (_dictionaryWords, _signalSize);

	/*

	CUBLAS_CHECK(cublasSetVector(_signalSize, sizeof(float), _hostSignal.data(), 1, _signal.get(), 1));
	CUDA_CHECK(cudaMemcpy(_dictionaryInverse.get(), _hostDictionaryInverse.data(), sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(_dictionary.get(), _hostDictionary.data(), sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));

	const float alphaScalar = 1.0f;
	const float negAlphaScalar = -1.0f;
	const float betaScalar = 0.0f;
	CUBLAS_CHECK(cublasSgemv(_cublasHandle, CUBLAS_OP_T, _signalSize, _dictionaryWords, &alphaScalar, _dictionaryInverse.get(), _signalSize, _signal.get(), 1, &betaScalar, _alpha.get(), 1));

	float signalNorm = 0.0f;
	CUBLAS_CHECK(cublasSnrm2(_cublasHandle, _signalSize, _signal.get(), 1, &signalNorm));
	float lambda = 1.0f / signalNorm;

	cuda_ptr<float> beta = make_cuda<float>(_dictionaryWords);
	cuda_ptr<float> interm = make_cuda<float>(_signalSize);

	dim3 blockSize(128);
	dim3 gridSize(_dictionaryWords + blockSize.x - 1 / blockSize.x);

	int iteration = 0;
	for (; iteration < iterations; iteration++)
	{
		// First we save the current alpha as the old one, in order to use it later
		CUBLAS_CHECK(cublasScopy(_cublasHandle, _dictionaryWords, _alpha.get(), 1, _alphaOld.get(), 1));

		GetBetaKrnl << <gridSize, blockSize >> > (lambda, _alpha.get(), beta.get(), _dictionaryWords);

		CUBLAS_CHECK(cublasSgemv(_cublasHandle, CUBLAS_OP_T, _dictionaryWords, _signalSize, &alphaScalar, _dictionary.get(), _dictionaryWords, beta.get(), 1, &betaScalar, interm.get(), 1));
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _signalSize, &negAlphaScalar, _signal.get(), 1, interm.get(), 1));

		CUBLAS_CHECK(cublasSgemv(_cublasHandle, CUBLAS_OP_T, _signalSize, _dictionaryWords, &alphaScalar, _dictionaryInverse.get(), _signalSize, interm.get(), 1, &betaScalar, _alpha.get(), 1));
		// axpy takes a single input/out parameter so we have to negate our subtraction and then multiply the result by -1.0 later
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _dictionaryWords, &negAlphaScalar, beta.get(), 1, _alpha.get(), 1));
		CUBLAS_CHECK(cublasSscal(_cublasHandle, _dictionaryWords, &negAlphaScalar, _alpha.get(), 1));

		ThresholdKrnl << <gridSize, blockSize >> > (_alpha.get(), _dictionaryWords, _alphaElementTh);
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _dictionaryWords, &negAlphaScalar, _alpha.get(), 1, _alphaOld.get(), 1));

		lambda = lambda * gamma;

		float diffNorm = 0.0f;
		CUBLAS_CHECK(cublasSnrm2(_cublasHandle, _dictionaryWords, _alphaOld.get(), 1, &diffNorm));
		if (diffNorm < _epsilon) {
			// We are done with the iterations. Norm is very small
			break;
		}
	}

	*/
}

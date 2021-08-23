#include "DeviceLiMapSCuBlas.cuh"

#include "cublas_shared.h"
#include "kernels/misc.cuh"
#include "kernels/threshold.cuh"

DeviceLiMapSCuBlas::DeviceLiMapSCuBlas(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize)
	: BaseLiMapS(solution, signal, D, DINV, dictionaryWords, signalSize)
{
	// We create our cublas handle 
	CUBLAS_CHECK(cublasCreate(&_cublasHandle));

	// We prepare all the GPU memory space
	_alphaH.resize(_dictionaryWords);

	_solution = make_cuda<float>(_dictionaryWords);
	_signal = make_cuda<float>(_signalSize);
	_dictionary = make_cuda<float>(_dictionaryWords * _signalSize);
	_dictionaryInverse = make_cuda<float>(_dictionaryWords * _signalSize);

	_alpha = make_cuda<float>(_dictionaryWords);
	_alphaNew = make_cuda<float>(_dictionaryWords);

}

DeviceLiMapSCuBlas::~DeviceLiMapSCuBlas()
{
	CUBLAS_CHECK(cublasDestroy(_cublasHandle));
}

void DeviceLiMapSCuBlas::Execute(int iterations)
{
	// We load our data
	CUBLAS_CHECK(cublasSetVector(_signalSize, sizeof(float), _signalHost, 1, _signal.get(), 1));
	CUDA_CHECK(cudaMemcpy(_dictionaryInverse.get(), _dictionaryInverseHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(_dictionary.get(), _dictionaryHost, sizeof(float) * _dictionaryWords * _signalSize, cudaMemcpyHostToDevice));

	// We calculate the initial alpha
	const float alphaScalar = 1.0f;
	const float negAlphaScalar = -1.0f;
	const float betaScalar = 0.0f;
	CUBLAS_CHECK(cublasSgemv(_cublasHandle, CUBLAS_OP_T, _signalSize, _dictionaryWords, &alphaScalar, _dictionaryInverse.get(), _signalSize, _signal.get(), 1, &betaScalar, _alpha.get(), 1));

	// We calculate the signal norm to derive the first lambda
	float signalNorm = 0.0f;
	CUBLAS_CHECK(cublasSnrm2(_cublasHandle, _signalSize, _signal.get(), 1, &signalNorm));
	float lambda = 1.0f / signalNorm;

	cuda_ptr<float> beta = make_cuda<float>(_dictionaryWords);
	cuda_ptr<float> interm = make_cuda<float>(_signalSize);

	// We pre-calculate our grid and block size (considering an unrolling factor of 8) that will be used with our custom kernel
	dim3 blockSize(128);
	dim3 gridSize = GetGridSize(blockSize, _dictionaryWords, 8);

	int iteration = 0;
	for (; iteration < iterations; iteration++)
	{
		// First we save the current alpha as the old one, in order to use it later
		CUBLAS_CHECK(cublasScopy(_cublasHandle, _dictionaryWords, _alpha.get(), 1, _alphaNew.get(), 1));

		// We calculate the beta vector 
		CalculateBeta<8> << <gridSize, blockSize >> > (_alpha.get(), beta.get(), lambda, _dictionaryWords);

		// 3.2) We need to compute the intermediate (dic * beta - sig) vector
		CUBLAS_CHECK(cublasSgemv(_cublasHandle, CUBLAS_OP_T, _dictionaryWords, _signalSize, &alphaScalar, _dictionary.get(), _dictionaryWords, beta.get(), 1, &betaScalar, interm.get(), 1));
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _signalSize, &negAlphaScalar, _signal.get(), 1, interm.get(), 1));

		// 3.3) We compute the new alpha with the thresholding at the end
		CUBLAS_CHECK(cublasSgemv(_cublasHandle, CUBLAS_OP_T, _signalSize, _dictionaryWords, &alphaScalar, _dictionaryInverse.get(), _signalSize, interm.get(), 1, &betaScalar, _alpha.get(), 1));
		// axpy takes a single input/out parameter so we have to negate our subtraction and then multiply the result by -1.0 later
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _dictionaryWords, &negAlphaScalar, beta.get(), 1, _alpha.get(), 1));
		CUBLAS_CHECK(cublasSscal(_cublasHandle, _dictionaryWords, &negAlphaScalar, _alpha.get(), 1));

		ThresholdVector<8> << <gridSize, blockSize >> > (_alpha.get(), _dictionaryWords);

		// We calculate the difference between the two alphas to later calculate the diff vector norm
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _dictionaryWords, &negAlphaScalar, _alpha.get(), 1, _alphaNew.get(), 1));

		lambda = lambda * gamma;

		float diffNorm = 0.0f;
		CUBLAS_CHECK(cublasSnrm2(_cublasHandle, _dictionaryWords, _alphaNew.get(), 1, &diffNorm));
		if (diffNorm < _epsilon) {
			// We are done with the iterations. Norm is very small
			break;
		}
	}

	printf("CuBlas iterations: %d\n", iteration);
	CUDA_CHECK(cudaMemcpy(_alphaH.data(), _alpha.get(), sizeof(float) * _dictionaryWords, cudaMemcpyDeviceToHost));

}

#include "DeviceLiMapSv1.cuh"

#include "cublas_shared.h"
#include "kernels.cuh"
#include "threshold_kernels.cuh"

DeviceLiMapSv1::DeviceLiMapSv1(std::vector<float>& solution, std::vector<float>& signal, std::vector<float>& D, std::vector<float>& DINV)
	:_signalSize(signal.size()), _dictionaryWords(solution.size()),
	// To avoid C++ vector copies, let's just store the vector references for our input data. This may be dangerous since the class MUST have the same (or shorted)
	// scope of our data, but for our purposes should be ok
	_hostSolution(solution), _hostSignal(signal), _hostDictionary(D), _hostDictionaryInverse(DINV)
{
	CUBLAS_CHECK(cublasCreate(&_cublasHandle));

	_solution = make_cuda<float>(solution.size());
	_signal = make_cuda<float>(signal.size());
	_dictionary = make_cuda<float>(D.size());
	_dictionaryInverse = make_cuda<float>(DINV.size());

	_alpha = make_cuda<float>(solution.size());
	_alphaNew = make_cuda<float>(solution.size());

}

DeviceLiMapSv1::~DeviceLiMapSv1()
{
	CUBLAS_CHECK(cublasDestroy(_cublasHandle));
}

void DeviceLiMapSv1::Execute(int iterations)
{
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
		CUBLAS_CHECK(cublasScopy(_cublasHandle, _dictionaryWords, _alpha.get(), 1, _alphaNew.get(), 1));

		GetBetaKrnl << <gridSize, blockSize >> > (lambda, _alpha.get(), beta.get(), _dictionaryWords);

		CUBLAS_CHECK(cublasSgemv(_cublasHandle, CUBLAS_OP_T, _dictionaryWords, _signalSize, &alphaScalar, _dictionary.get(), _dictionaryWords, beta.get(), 1, &betaScalar, interm.get(), 1));
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _signalSize, &negAlphaScalar, _signal.get(), 1, interm.get(), 1));

		CUBLAS_CHECK(cublasSgemv(_cublasHandle, CUBLAS_OP_T, _signalSize, _dictionaryWords, &alphaScalar, _dictionaryInverse.get(), _signalSize, interm.get(), 1, &betaScalar, _alpha.get(), 1));
		// axpy takes a single input/out parameter so we have to negate our subtraction and then multiply the result by -1.0 later
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _dictionaryWords, &negAlphaScalar, beta.get(), 1, _alpha.get(), 1));
		CUBLAS_CHECK(cublasSscal(_cublasHandle, _dictionaryWords, &negAlphaScalar, _alpha.get(), 1));

		ThresholdKrnl << <gridSize, blockSize >> > (_alpha.get(), _dictionaryWords, _alphaElementTh);
		CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _dictionaryWords, &negAlphaScalar, _alpha.get(), 1, _alphaNew.get(), 1));		

		lambda = lambda * gamma;

		float diffNorm = 0.0f;
		CUBLAS_CHECK(cublasSnrm2(_cublasHandle, _dictionaryWords, _alphaNew.get(), 1, &diffNorm));
		if (diffNorm < _epsilon) {
			// We are done with the iterations. Norm is very small
			break;
		}
	}
}

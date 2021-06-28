#include "DeviceLiMapSv1.h"
#include <cublas_api.h>

DeviceLiMapSv1::DeviceLiMapSv1(std::vector<float>& solution, std::vector<float>& signal, std::vector<float>& D, std::vector<float>& DINV)
	:_signalSize(signal.size()), _dictionaryWords(solution.size())
{
	CUBLAS_CHECK(cublasCreate(&_cublasHandle));

	_solution = make_cuda<float>(solution.size());
	_signal = make_cuda<float>(signal.size());
	_dictionary = make_cuda<float>(D.size());
	_dictionaryInverse = make_cuda<float>(DINV.size());

	_alpha = make_cuda<float>(solution.size());
	_alphaOld = make_cuda<float>(solution.size());

}

DeviceLiMapSv1::~DeviceLiMapSv1()
{
	CUBLAS_CHECK(cublasDestroy(_cublasHandle));
}

void DeviceLiMapSv1::Execute(int iterations)
{
	float neg = -1.0f;
	CUBLAS_CHECK(cublasSaxpy(_cublasHandle, _dictionaryWords, &neg, _alpha.get(), 1, _alphaOld.get(), 1));

	float norm = 1.0f;
	cublasSnrm2(_cublasHandle, _dictionaryWords, _alphaOld.get(), 1, &norm);

	__debugbreak();
}

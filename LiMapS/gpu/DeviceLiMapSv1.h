#pragma once

#include "cuda_shared.h"
#include "cublas_shared.h"
#include <vector>

class DeviceLiMapSv1
{
private:
	const int _signalSize;
	const int _dictionaryWords;

	cuda_ptr<float> _solution;
	cuda_ptr<float> _signal;
	cuda_ptr<float> _dictionary;
	cuda_ptr<float> _dictionaryInverse;

	cuda_ptr<float> _alpha;
	cuda_ptr<float> _alphaOld;

	cublasHandle_t _cublasHandle;

	const float epsilon = 1e-5f;
	const float _alphaElementTh = 1e-4f;
	const float gamma = 1.01f;
public:
	DeviceLiMapSv1(std::vector<float>& solution, std::vector<float>& signal, std::vector<float>& D, std::vector<float>& DINV);
	virtual ~DeviceLiMapSv1();

	void Execute(int iterations);
};


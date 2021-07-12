#pragma once

#include "cuda_shared.h"
#include "cublas_shared.h"
#include <vector>

class DeviceLiMapSv1
{
private:
	const size_t _signalSize;
	const size_t _dictionaryWords;

	std::vector<float>& _hostSolution;
	std::vector<float>& _hostSignal;
	std::vector<float>& _hostDictionary;
	std::vector<float>& _hostDictionaryInverse;

	cuda_ptr<float> _solution;
	cuda_ptr<float> _signal;
	cuda_ptr<float> _dictionary;
	cuda_ptr<float> _dictionaryInverse;

	cuda_ptr<float> _alpha;
	cuda_ptr<float> _alphaNew;

	cublasHandle_t _cublasHandle;

	const float _epsilon = 1e-5f;
	const float _alphaElementTh = 1e-4f;
	const float gamma = 1.01f;
public:
	DeviceLiMapSv1(std::vector<float>& solution, std::vector<float>& signal, std::vector<float>& D, std::vector<float>& DINV);
	virtual ~DeviceLiMapSv1();

	void Execute(int iterations);
};


#pragma once

#include "cuda_shared.h"
#include "cublas_shared.h"
#include "..\BaseLiMapS.h"
#include <vector>

class DeviceLiMapSCuBlas : public BaseLiMapS
{
private:
	cuda_ptr<float> _solution;
	cuda_ptr<float> _signal;
	cuda_ptr<float> _dictionary;
	cuda_ptr<float> _dictionaryInverse;

	cuda_ptr<float> _alpha;
	cuda_ptr<float> _alphaNew;

	std::vector<float> _alphaH;

	cublasHandle_t _cublasHandle;

	const float _epsilon = 1e-5f;
	const float _alphaElementTh = 1e-4f;
	const float gamma = 1.01f;
public:
	DeviceLiMapSCuBlas(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize);
	~DeviceLiMapSCuBlas();

	void Execute(int iterations);

	const std::vector<float>& GetAlpha() const { return _alphaH; };
};


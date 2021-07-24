#pragma once

#include "cuda_shared.h"
#include "cublas_shared.h"
#include "cublas_shared.h"
#include "../BaseLiMapS.h"
#include <vector>

class DeviceLiMapSv4 : public BaseLiMapS
{
private:
	std::vector<float> _alphaH;

	cuda_ptr<float> _solutionPtr;
	cuda_ptr<float> _signalPtr;
	cuda_ptr<float> _dictionaryPtr;
	cuda_ptr<float> _dictionaryInversePtr;

	cuda_ptr<float> _alphaPtr;
	cuda_ptr<float> _alphaOldPtr;

public:
	DeviceLiMapSv4(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize);

	void Execute(int iterations);

	inline const std::vector<float>& GetAlpha() const { return _alphaH; }
};

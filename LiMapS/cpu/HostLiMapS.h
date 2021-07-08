#pragma once

#include <vector>

#include "../BaseLiMapS.h"
#include "vectors.hpp"
#include "matrices.hpp"
#include "intrin_ext.h"


class HostLiMapS : public BaseLiMapS {
private:
	std::vector<float> _alpha;
	std::vector<float> _oldAlpha;

	std::vector<float> _snrs;


	void GetBeta(float* beta, float lambda);
public:
	HostLiMapS(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize);

	void Execute(int iterations);

	inline const std::vector<float>& GetAlpha() const { return _alpha; }

};
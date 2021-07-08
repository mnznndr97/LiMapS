#pragma once

#include <vector>

#include "vectors.hpp"
#include "matrices.hpp"
#include "intrin_ext.h"


class HostLiMapS {
private:
	std::vector<float>& _solution;
	std::vector<float>& _signal;
	std::vector<float>& _dictionary;
	std::vector<float>& _dictionaryInverse;

	std::vector<float> _alpha;
	std::vector<float> _oldAlpha;

	std::vector<float> _snrs;


	const float epsilon = 1e-5f;
	const float _alphaElementTh = 1e-4f;
	const float gamma = 1.01f;

	void GetBeta(float* beta, float lambda);
public:
	HostLiMapS(std::vector<float>& solution, std::vector<float>& signal, std::vector<float>& D, std::vector<float>& DINV);

	void Execute(int iterations);

	inline const std::vector<float>& GetAlpha() const { return _alpha; }

};
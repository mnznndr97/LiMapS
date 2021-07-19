#include "HostLiMapS.h"
#include "vectors.hpp"

HostLiMapS::HostLiMapS(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize)
	: BaseLiMapS(solution, signal, D, DINV, dictionaryWords, signalSize)
{
	// Let's pre-allocate the alpha and the old alpha space
	_alpha.resize(dictionaryWords);
	_newAlpha.resize(dictionaryWords);
}

void HostLiMapS::GetBeta(float* beta, float lambda) {
	// beta is calulated by appling the reduction function specified in the paper

	size_t index = 0;

	__m256 nLambdaV = _mm256_set1_ps(-lambda);
	__m256 onesV = _mm256_set1_ps(1.0f);

	// beta is same size al alpha
	for (index = 0; index < _alpha.size() / 8; index++)
	{
		__m256 alphaV = _mm256_load_ps(&_alpha[index * 8]);

		__m256 data = _mm256_abs_ps(alphaV);
		data = _mm256_mul_ps(nLambdaV, data);
		data = _mm256_sub_ps(onesV, _mm256_exp_ps(data));

		__m256 betaV = _mm256_mul_ps(alphaV, data);
		_mm256_store_ps(&beta[index * 8], betaV);
	}

	for (size_t i = index * 8; i < _alpha.size(); i++)
	{
		float alphaD = _alpha[i];

		float data = -lambda * fabsf(alphaD);
		data = 1.0f - expf(data);

		beta[i] = alphaD * data;
	}
}

void HostLiMapS::Execute(int iterations)
{
	// The first thing we need to do is to calculate the starting alpha
	Mat2VecProduct(_dictionaryInverseHost, _dictionaryWords, _signalSize, _signalHost, _alpha.data());
	// We set the first "old" alpha equals to the "new" one
	std::copy(_alpha.begin(), _alpha.end(), _newAlpha.begin());

	// Then we calculate the first lambda value using the signal norm
	float signalNorm = GetEuclideanNorm(_signalHost, _signalSize);
	float lambda = 1.0f / signalNorm;

	// We need two temporary arrays to do our job
	std::vector<float> beta(_alpha.size());
	std::vector<float> interm(_signalSize);

	// Let's start with our loop
	int iteration = 0;
	for (; iteration < iterations; iteration++)
	{
		// Swap the alpha vector, so we don't have to perform any mem copy
		std::swap(_alpha, _newAlpha);

		GetBeta(beta.data(), lambda);

		// D * beta - s
		Mat2VecProduct(_dictionaryHost, _signalSize, _dictionaryWords, beta.data(), interm.data());
		SubVec(interm.data(), _signalHost, interm.data(), _signalSize);

		// beta - DINV * (D * beta - s);
		Mat2VecProduct(_dictionaryInverseHost, _dictionaryWords, _signalSize, interm.data(), _newAlpha.data());
		SubVec(beta.data(), _newAlpha.data(), _newAlpha.data(), _dictionaryWords);

		// Alpha thresholding 
		ThresholdVec(_newAlpha.data(), _dictionaryWords, _alphaElementTh);
		lambda = gamma * lambda;

		if (GetDiffEuclideanNorm(_newAlpha.data(), _alpha.data(), _dictionaryWords) < epsilon) {
			// We are done with the iterations. Norm is very small
			break;
		}
	}

	printf("CPU iterations: %d\n", iteration);
}

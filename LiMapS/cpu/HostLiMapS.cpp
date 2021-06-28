#include "HostLiMapS.h"
#include "matrices.hpp"
#include "vectors.hpp"

HostLiMapS::HostLiMapS(std::vector<float>& solution, std::vector<float>& signal, std::vector<float>& D, std::vector<float>& DINV)
// To avoid C++ vector copies, let's just store the vector references for our input data. This may be dangerous since the class MUST have the same (or shorted)
// scope of our data, but for our purposes should be ok
	: _solution(solution), _signal(signal), _dictionary(D), _dictionaryInverse(DINV)
{
	// Let's pre-allocate the alpha and the old alpha space
	_alpha.resize(solution.size());
	_oldAlpha.resize(solution.size());
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

		float data = -lambda * fabs(alphaD);
		data = 1.0f - exp(data);

		beta[i] = alphaD * data;
	}
}

void HostLiMapS::Execute(int iterations)
{
	// The first thing we need to do is to calculate the starting alpha
	Mat2VecProduct(_dictionaryInverse.data(), _solution.size(), _signal.size(), _signal.data(), _alpha.data());

	float signalNorm = GetEuclideanNorm(_signal);
	float lambda = 1.0f / signalNorm;

	// We need two temporary arrays to do out job
	std::vector<float> beta(_alpha.size());
	std::vector<float> interm(_signal.size());

	// Let's start with our loop
	int iteration = 0;
	for (; iteration < iterations; iteration++)
	{
		// First we save the current alpha as the old one, in order to use it later
		std::copy(_alpha.begin(), _alpha.end(), _oldAlpha.begin());

		// beta = alpha.*(1 - exp(-lambda.*abs(alpha)));
		GetBeta(beta.data(), lambda);

		Mat2VecProduct(_dictionary.data(), _signal.size(), _solution.size(), beta.data(), interm.data());
		SubVec(interm, _signal, interm);

		Mat2VecProduct(_dictionaryInverse.data(), _solution.size(), _signal.size(), interm.data(), _alpha.data());
		SubVec(beta, _alpha, _alpha);

		ThresholdVec(_alpha, _alphaElementTh);
		lambda = gamma * lambda;

		if (GetDiffEuclideanNorm(_alpha, _oldAlpha) < epsilon) {
			// We are done with the iterations. Norm is very small
			break;
		}
	}
}

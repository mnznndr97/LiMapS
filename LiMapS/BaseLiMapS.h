#pragma once

#include <vector>

class BaseLiMapS
{
protected:
	const size_t _dictionaryWords;
	const size_t _signalSize;

	const float* _solutionHost;
	const float* _signalHost;
	const float* _dictionaryHost;
	const float* _dictionaryInverseHost;


	const float epsilon = 1e-5f;
	const float _alphaElementTh = 1e-4f;
	const float gamma = 1.01f;

public:
	inline BaseLiMapS(const float* solution, const float* signal, const float* D, const float* DINV, size_t dictionaryWords, size_t signalSize) :
		_dictionaryWords(dictionaryWords), _signalSize(signalSize),
		// To avoid C++ vector copies, let's just store the vector references for our input data. This may be dangerous since the class MUST have the same (or shorted)
		// scope of our data, but for our purposes should be ok
		_solutionHost(solution), _signalHost(signal), _dictionaryHost(D), _dictionaryInverseHost(DINV)
	{

	}

	virtual ~BaseLiMapS() {
	}

	virtual void Execute(int iterations) = 0;
	virtual const std::vector<float>& GetAlpha() const = 0;
};
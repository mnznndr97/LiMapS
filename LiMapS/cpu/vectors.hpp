#pragma once
#include <memory>
#include <assert.h>
#include <intrin.h>

float GetEuclideanNorm(const float* vector, size_t size);
float GetEuclideanNorm(const std::unique_ptr<float[]>& vector, size_t size);
float GetEuclideanNorm(const std::vector<float>& vector);

float GetEuclideanNorm(const std::vector<float>& vector) {
	return GetEuclideanNorm(vector.data(), vector.size());
}

float GetEuclideanNorm(const std::unique_ptr<float[]>& vector, size_t size) {
	return GetEuclideanNorm(vector.get(), size);
}

float GetEuclideanNorm(const float* vector, size_t size) {
	// We assumes that our data are correct
	assert(vector != nullptr);
	assert(size >= 0);

	if (size == 0) return 0.0f;

	__m256 sumVector = _mm256_setzero_ps();
	float totalSum = 0.0f;
	size_t index = 0;
	for (index = 0; index < size / 8; index++)
	{
		__m256 data = _mm256_load_ps(vector + (index * 8));
		sumVector = _mm256_add_ps(sumVector, _mm256_mul_ps(data, data));
	}

	for (int i = index * 8; i < size; i++) {
		totalSum += vector[i] * vector[i];
	}

	// Too "tricky" to compute the hadd for the 256 bit vector
	// Let's just copy it to a temp array for the moment
	float temp[8];
	_mm256_store_ps(temp, sumVector);
	for (size_t i = 0; i < 8; i++)
	{
		totalSum += temp[i];
	}


	return sqrt(totalSum);
}
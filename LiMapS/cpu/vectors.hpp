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
		sumVector = _mm256_fmadd_ps(data, data, sumVector);
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

float GetDotProduct(const float* v1, const float* v2, size_t size) {
	// We assumes that our data are correct
	assert(v1 != nullptr);
	assert(v2 != nullptr);
	assert(size >= 0);

	if (size == 0) return 0.0f;

	__m256 dpSums = _mm256_setzero_ps();
	size_t index = 0;
	for (index = 0; index < size / 8; index++)
	{
		__m256 vec1 = _mm256_load_ps(v1 + (index * 8));
		__m256 vec2 = _mm256_load_ps(v2 + (index * 8));
		__m256 dotProduct = _mm256_dp_ps(vec1, vec2, 0xF1);

		dpSums = _mm256_add_ps(dpSums, dotProduct);
	}

	float temp[5];
	_mm256_maskstore_ps(temp, _mm256_set_epi32(0, 0, 0, 0x80000000, 0, 0, 0, 0x80000000), dpSums);

	float dotProd = temp[0] + temp[4];
	for (size_t i = index * 8; i < size; i++)
	{
		dotProd += v1[i] * v2[i];
	}

	return dotProd;
}
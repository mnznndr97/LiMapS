#include "vectors.hpp"

#include <intrin.h>
#include "intrin_ext.h"

void SubVec(const float* a, const float* b, float* dest, size_t size) {
	// Simple subtraction loop using first a 8 floats pack register
	// then a naive single item loop
	
	size_t index = 0;
	for (index = 0; index < size / 8; index++)
	{
		__m256 aVec = _mm256_load_ps(a + (index * 8));
		__m256 bVec = _mm256_load_ps(b + (index * 8));

		_mm256_store_ps(dest + (index * 8), _mm256_sub_ps(aVec, bVec));
	}

	for (size_t i = index * 8; i < size; i++)
	{
		dest[i] = a[i] - b[i];
	}
}

float GetEuclideanNorm(const float* vector, size_t size) {
	// We assumes that our data are correct
	assert(vector != nullptr);
	assert(size >= 0);

	if (size == 0) return 0.0f;

	__m256 sumVector = _mm256_setzero_ps();
	size_t index = 0;
	for (index = 0; index < size / 8; index++)
	{
		__m256 data = _mm256_load_ps(vector + (index * 8));
		sumVector = _mm256_fmadd_ps(data, data, sumVector);
	}

	float tailSum = 0.0f;
	for (size_t i = index * 8; i < size; i++) {
		tailSum += vector[i] * vector[i];
	}

	return sqrtf(tailSum + __mm256_sumall_ps(sumVector));
}

float GetDiffEuclideanNorm(const float* vector1, const float* vector2, size_t size) {
	assert(vector1 != nullptr);
	assert(vector2 != nullptr);
	assert(size >= 0);

	if (size == 0) return 0.0f;

	__m256 sumVector = _mm256_setzero_ps();
	size_t index = 0;
	for (index = 0; index < size / 8; index++)
	{
		// Simple data load, diff, square and cumulative sum here
		__m256 data1 = _mm256_load_ps(vector1 + (index * 8));
		__m256 data2 = _mm256_load_ps(vector2 + (index * 8));

		__m256 sub = _mm256_sub_ps(data1, data2);
		sumVector = _mm256_fmadd_ps(sub, sub, sumVector);
	}

	float tailSum = 0.0f;
	for (size_t i = index * 8; i < size; i++) {
		float data = vector1[i] - vector2[i];
		tailSum += data * data;
	}

	return sqrtf(tailSum + __mm256_sumall_ps(sumVector));
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
		// Simple data load and packed dot product here
		__m256 vec1 = _mm256_load_ps(v1 + (index * 8));
		__m256 vec2 = _mm256_load_ps(v2 + (index * 8));
		__m256 dotProduct = _mm256_dp_ps(vec1, vec2, 0xF1);

		dpSums = _mm256_add_ps(dpSums, dotProduct);
	}

	float tailDotProd = 0.0f;
	for (size_t i = index * 8; i < size; i++)
	{
		tailDotProd += v1[i] * v2[i];
	}

	return tailDotProd + __mm256_sumall_ps(dpSums);
}

/// <summary>
/// Set to 0 all the element below the specified threshold
/// </summary>
void ThresholdVec(float* vec, size_t size, float threshold) {
	size_t index = 0;
	for (index = 0; index < size / 8; index++)
	{
		__m256 data = _mm256_load_ps(vec + (index * 8));
		__m256 mask = _mm256_cmp_ps(_mm256_abs_ps(data), _mm256_set1_ps(threshold), _CMP_GE_OQ);
		// We need to conver the mask to a "real" float register. (Data are stored as -1 but in int format)
		__m256i ints = _mm256_castps_si256(mask);
		mask = _mm256_cvtepi32_ps(ints);

		// Little "normalization" trick here: with IEEE754 we can have "negative" zero. This should be no 
		// problem since they should test equals but to "fix" we can simply add a all zero vector
		data = _mm256_fmadd_ps(data, _mm256_abs_ps(mask), _mm256_setzero_ps());
		_mm256_store_ps(vec + (index * 8), data);
	}

	for (size_t i = index * 8; i < size; i++)
	{
		if (vec[i] < threshold) {
			vec[i] = 0.0f;
		}
	}
}

void Mat2VecProduct(const float* matrix, size_t rows, size_t cols, const float* colVector, float* destination)
{
	assert(matrix != nullptr);
	assert(colVector != nullptr);
	assert(destination != nullptr);
	assert(rows > 0);
	assert(cols > 0);

	for (size_t row = 0; row < rows; row++)
	{
		const float* rowVector = &matrix[row * cols];
		destination[row] = GetDotProduct(rowVector, colVector, cols);
	}
}

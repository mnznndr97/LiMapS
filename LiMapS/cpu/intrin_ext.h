#pragma once

/**
* Intel Intrinsics extension header
*/

#include <intrin.h>

/// <summary>
/// Returns the absolute value for a packed floating point vector
/// </summary>
inline __m256 _mm256_abs_ps(__m256 v){
	// To get the abs for a float vector, we have simply to "clear" the sign bit in the float representation
	// In order to do this, we load the mask "0x7FFFFFF" (8times) in a 256 vector and we bitwise and everything 
	__m256 absMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
	return _mm256_and_ps(v, absMask);
}

/// <summary>
/// Sums all the packed float elements of a vector
/// </summary>
inline float __mm256_sumall_ps(__m256 v) {
	__m256 haddTurn = _mm256_hadd_ps(v, _mm256_permute2f128_ps(v, _mm256_setzero_ps(), 0x11));
	haddTurn = _mm256_hadd_ps(haddTurn, _mm256_setzero_ps());
	haddTurn = _mm256_hadd_ps(haddTurn, _mm256_setzero_ps());

	return _mm256_cvtss_f32(haddTurn);
}
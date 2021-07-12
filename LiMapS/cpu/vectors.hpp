#pragma once
#include <memory>
#include <assert.h>

/// <summary>
/// Calculates the euclidean norm of a vector
/// </summary>
float GetEuclideanNorm(const float* vector, size_t size);

/// <summary>
/// Calculates the euclidean norm of the difference between 2 vectors
/// </summary>
float GetDiffEuclideanNorm(const float* vector1, const float* vector2, size_t size);

/// <summary>
/// Calculates the dot product between two vectors
/// </summary>
float GetDotProduct(const float* v1, const float* v2, size_t size);

/// <summary>
/// Subtracts two vector into a third destination vector
/// </summary>
void SubVec(const float* a, const float* b, float* dest, size_t size);

/// <summary>
/// Sets all the values 
/// </summary>
void ThresholdVec(float* vec, size_t size, float threshold);

/// <summary>
/// Performs a matrix-vector multiplication
/// </summary> 
void Mat2VecProduct(const float* matrix, size_t rows, size_t cols, const float* colVector, float* destination);
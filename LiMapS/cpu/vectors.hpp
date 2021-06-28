#pragma once
#include <memory>
#include <assert.h>
#include <intrin.h>
#include <vector>
#include <span>

void AddVec(const float* a, const float* b, float* dest, size_t size);
float GetEuclideanNorm(const float* vector, size_t size);
float GetEuclideanNorm(const std::vector<float>& vector);
float GetEuclideanNorm(std::unique_ptr<float[]> vector, size_t size);

/// <summary>
/// Calculates the euclideam norm of the difference between 2 vectors
/// </summary>
float GetDiffEuclideanNorm(const std::vector<float>& vector1, const std::vector<float>& vector2);
/// <summary>
/// Calculates the euclideam norm of the difference between 2 vectors
/// </summary>
float GetDiffEuclideanNorm(const float* vector1, const float* vector2, size_t size);

float GetDotProduct(const float* v1, const float* v2, size_t size);
void SubVec(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& dest);
void SubVec(const float* a, const float* b, float* dest, size_t size);

void ThresholdVec(std::vector<float>& vec, float threshold);
void ThresholdVec(float* vec, size_t size, float threshold);


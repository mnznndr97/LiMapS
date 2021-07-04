#pragma once

#include "cuda_shared.h"


/// <summary>
/// Sums all the data relatively to a warp
/// </summary>
/// <param name="data">Warp data</param>
/// <returns>Sums of the </returns>
__inline__ __device__ float WarpReduce(float data);

__global__ void Norm(const float* vec, size_t size, float* result);
__global__ void NormDiff(const float* vec1, const float* vec2, size_t size, float* result);
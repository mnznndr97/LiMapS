#pragma once

#include "cuda_shared.h"

__device__ float GetBeta(float lambda, float data);

__global__ void GetBetaKrnl(float lambda, const float* data, float* beta, size_t size);


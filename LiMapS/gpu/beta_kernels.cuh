#pragma once

#include "cuda_shared.h"
__global__ void GetBetaKrnl(float lambda, const float* data, float* beta, size_t size);


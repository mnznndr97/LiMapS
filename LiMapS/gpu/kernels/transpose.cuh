#pragma once

#include "..\cuda_shared.h"
#include <nvfunctional>

#define TILE_DIM    16
#define BLOCK_ROWS  16

/// <summary>
/// Transpose that effectively reorders execution of thread blocks along diagonals of the
/// matrix (also coalesced and has no bank conflicts)
///
/// Here blockIdx.x is interpreted as the distance along a diagonal and blockIdx.y as
/// corresponding to different diagonals
///
/// blockIdx_x and blockIdx_y expressions map the diagonal coordinates to the more commonly
/// used cartesian coordinates so that the only changes to the code from the coalesced version
/// are the calculation of the blockIdx_x and blockIdx_y and replacement of blockIdx.x and
/// bloclIdx.y with the subscripted versions in the remaining code
/// </summary>
/// <remarks>
/// Code from Cuda samples
/// </remarks>
__global__ void Transpose(const float* __restrict__ source, float* destination, size_t width, size_t height);

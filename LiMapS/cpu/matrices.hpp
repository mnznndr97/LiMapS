#pragma once

#include <assert.h>
#include <algorithm>

#include "vectors.hpp"

void Mat2VecProduct(const float* matrix, size_t rows, size_t cols, const float* colVector, size_t size, float* destination) {
	assert(matrix != nullptr);
	assert(colVector != nullptr);
	assert(destination != nullptr);
	assert(rows > 0);
	assert(cols > 0);
	assert(size == cols);

	for (size_t row = 0; row < rows; row++)
	{
		const float* rowVector = &matrix[row * cols];
		destination[row] = GetDotProduct(rowVector, colVector, cols);
	}
}
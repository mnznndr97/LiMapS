#include "kernels.cuh"

__global__ void SquareSumKrnl(const float* vec, size_t size, float* result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums
	float data = idx < size ? vec[idx] : 0.0f;
	data = data * data;

	KernelReduce<float*>(data, size, [](float* dest, float sum) { atomicAdd(dest, sum); }, result);
}

__global__ void Matrix2Vector2B(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0f;
	// We iterate over block size to reach width. We iterate in blocks of a size of a warp
	// In this way vector items are read one time in a single transaction for each warp
	// NB: each thread is executing the operations, even if it is operating with an out-of-bound index;
	// in this case the value calculated will always be 0.0f
	for (size_t i = 0; i < ((width + warpSize - 1) / warpSize); i++)
	{
		size_t base = i * warpSize;
		size_t index = base + (threadIdx.x % warpSize);

		// Vector read: each thread in the warp reads it own items, contiguous to the others
		float data = vector[index];

		// After data read, each thread in the warp shares the vector value with the other threads to calculare the sum
#pragma unroll
		for (int w = 0; w < warpSize; w++)
		{
			float vectorData = __shfl_sync(FULL_MASK, data, w);

			index = base + w;
			// Each thread in the warp is reading the same item but in different rows. So better use a column-major ordering
			if ((idx < height) && (index < width)) {
				index = index * height + idx;
				sum += (matrix[index] * vectorData);
			}
		}
	}

	// Now if we are in our bound, we invoke the callback
	if (idx < height) {
		dest[idx] = sum;
	}
}

__global__ void Matrix2Vector3B(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float sData[128];

	bool inBounds = idx < height;
	int threadInWarp = threadIdx.x % warpSize;

	float sum = 0.0f;
	for (size_t b = 0; b < ((width + blockDim.x - 1) / blockDim.x); b++)
	{
		size_t index = b * blockDim.x + threadIdx.x;
		if (index < width) {
			sData[threadIdx.x] = vector[index];
		}
		else
		{
			sData[threadIdx.x] = 0.0f;
		}

		__syncthreads();

		for (size_t i = 0; i < blockDim.x / warpSize; i++)
		{
			index = b * blockDim.x + (i * warpSize);
			//float data = sData[i * warpSize + (threadIdx.x % warpSize)];

#pragma unroll
			for (int w = 0; w < warpSize; w++)
			{
				float vectorData = sData[i * warpSize + w];
				float matrixData = (inBounds && (index + w < width)) ? matrix[(index + w) * height + idx] : 0.0f;
				sum += matrixData * vectorData;
			}

		}
	}


	if (inBounds) {
		dest[idx] = sum;
	}
}

__global__ void Matrix2VectorPartitionB(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.0f;
	// We iterate over block size to reach width. We iterate in blocks of a size of a warp
	// In this way vector items are read one time in a single transaction for each warp
	// NB: each thread is executing the operations, even if it is operating with an out-of-bound index;
	// in this case the value calculated will always be 0.0f
	for (size_t i = 0; i < ((width + warpSize - 1) / warpSize); i++)
	{
		size_t base = i * warpSize;
		size_t index = base + (threadIdx.x % warpSize);

		// Vector read: each thread in the warp reads it own items, contiguous to the others
		float data = 0.0f;
		if (index < width) {
			index = (index + (blockDim.x * blockIdx.x));
			index = Mod(index, width);
			data = vector[index];
		}

		// After data read, each thread in the warp shares the vector value with the other threads to calculare the sum
#pragma unroll
		for (int w = 0; w < warpSize; w++)
		{
			float vectorData = __shfl_sync(FULL_MASK, data, w);
			float matrixData = 0.0f;
			// Each thread in the warp is reading the same item but in different rows. So better use a column-major ordering
			if ((idx < height) && (base + w < width)) {
				index = base + (blockDim.x * blockIdx.x) + w;
				index = Mod(index, width);
				index = index * height + idx;
				matrixData = matrix[index];
			}
			sum += matrixData * vectorData;
		}
	}

	// Now if we are in our bound, we invoke the callback
	if (idx < height) {
		dest[idx] = sum;
	}
}

__global__ void Transpose(const float* __restrict__ source, float* destination, size_t width, size_t height) {
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	size_t size = width * height;

	int blockIdx_x, blockIdx_y;

	// do diagonal reordering
	if (width == height)
	{
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	}
	else
	{
		int bid = blockIdx.x + gridDim.x * blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}

	// from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
	// and similarly for y

	int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	// Let's read all the tile rows. Remember that each block reads 
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		if (xIndex < width && yIndex < height)
			tile[threadIdx.y + i][threadIdx.x] = source[index_in + i * width];
	}

	cg::sync(cta);

	xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
	{
		if (xIndex < height && yIndex < width)
			destination[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}
}
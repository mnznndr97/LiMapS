#include "kernels.cuh"

__global__ void SquareSumKrnl(const float* vec, size_t size, float* result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// We calculate the squared value. We maintain the entire warp active but if we are out of bounds we
	// use zero as data value. In this way no sum error are introduced in the final sums
	float data = idx < size ? vec[idx] : 0.0f;
	data = data * data;

	KernelReduce<float*>(data, size, [](float* dest, float sum) { atomicAdd(dest, sum); }, result);
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
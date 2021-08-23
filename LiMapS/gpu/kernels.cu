#include "kernels.cuh"

__global__ void Matrix2VectorSharedMem(const float* __restrict__ matrix, const float* __restrict__ vector, float* dest, size_t width, size_t height) {
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
#include "matrix2vector.cuh"

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

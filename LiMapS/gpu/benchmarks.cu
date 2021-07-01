#include "benchmarks.cuh"

#include <iostream>

#include "cuda_shared.h"
#include "beta_kernels.cuh"
#include "threshold_kernels.cuh"

void RunKernelsBenchmarks() {
	std::cout << "Starting benchmarks" << std::endl;

	size_t dataSize = 80000;
	cuda_ptr<float> zeroArray = make_cuda<float>(dataSize);
	cuda_ptr<float> destArray = make_cuda<float>(dataSize);

	dim3 blockSize(32);
	GetBetaKrnl << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x >> > (1.0f, zeroArray.get(), destArray.get(), dataSize);	
	cudaDeviceSynchronize();

	blockSize.x = 64;
	GetBetaKrnl << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x >> > (1.0f, zeroArray.get(), destArray.get(), dataSize);
	cudaDeviceSynchronize();

	blockSize.x = 128;
	GetBetaKrnl << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x >> > (1.0f, zeroArray.get(), destArray.get(), dataSize);
	cudaDeviceSynchronize();

	blockSize.x = 256;
	GetBetaKrnl << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x >> > (1.0f, zeroArray.get(), destArray.get(), dataSize);
	cudaDeviceSynchronize();

	blockSize.x = 512;
	GetBetaKrnl << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x >> > (1.0f, zeroArray.get(), destArray.get(), dataSize);
	cudaDeviceSynchronize();
}
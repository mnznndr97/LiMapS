#include "benchmarks.cuh"

#include <iostream>

#include "cuda_shared.h"
#include "cublas_shared.h"
#include "kernels.cuh"
#include "beta_kernels.cuh"
#include "threshold_kernels.cuh"

__global__ void Fill(float* data, size_t size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		data[index] = 1.0f;
	}
}

void RunNormBenchmarks(size_t dataSize) {
	cuda_ptr<float> data = make_cuda<float>(dataSize);

	dim3 blockSize(256);
	Fill << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x >> > (data.get(), dataSize);

	float norm = 0.0f;
	std::cout << "Starting NORM kernel comparison benchmarks" << std::endl;
	cublasHandle_t cublasHandle;
	CUBLAS_CHECK(cublasCreate(&cublasHandle));
	cublasSnrm2(cublasHandle, dataSize, data.get(), 1, &norm);
	CUBLAS_CHECK(cublasDestroy(cublasHandle));

	std::cout << "Cublas norm: " << norm << std::endl;

	cuda_ptr<float> deviceNorm = make_cuda<float>(1);

	blockSize.x = 32;
	Norm << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel: " << norm << std::endl;

	blockSize.x = 64;
	Norm << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 128;
	Norm << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 256;
	Norm << <(dataSize + blockSize.x - 1) / blockSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());
}

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
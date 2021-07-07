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
	dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
	Fill << <gridSize, blockSize.x >> > (data.get(), dataSize);

	float norm = 0.0f;
	std::cout << "Starting NORM kernel comparison benchmarks" << std::endl;
	cublasHandle_t cublasHandle;
	CUBLAS_CHECK(cublasCreate(&cublasHandle));
	cublasSnrm2(cublasHandle, dataSize, data.get(), 1, &norm);
	CUBLAS_CHECK(cublasDestroy(cublasHandle));

	std::cout << "Cublas norm: " << norm << std::endl;

	cuda_ptr<float> deviceNorm = make_cuda<float>(1);

	blockSize.x = 32;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	SquareSumKrnl << <gridSize, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel: " << norm << std::endl;

	blockSize.x = 64;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	SquareSumKrnl << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 128;
	SquareSumKrnl << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 256;
	SquareSumKrnl << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());


	// 2-reduction kernels
	cudaMemset(deviceNorm.get(), CUBLAS_GEMM_ALGO0, sizeof(float));

	blockSize.x = 32;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	SquareSumKrnlUnroll<2><< <(gridSize.x +  1) / 2, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel 2: " << norm << std::endl;

	blockSize.x = 64;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	SquareSumKrnlUnroll<2> << <(gridSize.x + 1) / 2, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 128;
	SquareSumKrnlUnroll<2> << <(gridSize.x + 1) / 2, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 256;
	SquareSumKrnlUnroll<2> << <(gridSize.x + 1) / 2, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	// 8-reduction kernels
	cudaMemset(deviceNorm.get(), CUBLAS_GEMM_ALGO0, sizeof(float));

	blockSize.x = 64;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	gridSize.x = (gridSize.x + 7) / 8;
	SquareSumKrnlUnroll<8> << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel 8: " << norm << std::endl;

	blockSize.x = 128;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	gridSize.x = (gridSize.x + 7) / 8;
	SquareSumKrnlUnroll<8> << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 256;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	gridSize.x = (gridSize.x + 7) / 8;
	SquareSumKrnlUnroll<8> << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	

	// 16-reduction with cache kernels
	cudaMemset(deviceNorm.get(), CUBLAS_GEMM_ALGO0, sizeof(float));

	blockSize.x = 128;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	gridSize.x = (gridSize.x + 15) / 16;
	SquareSumKrnlUnrollLdg<16> << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel 16: " << norm << std::endl;

	blockSize.x = 256;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	gridSize.x = (gridSize.x + 15) / 16;
	SquareSumKrnlUnrollLdg<16> << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 512;
	gridSize.x = (dataSize + blockSize.x - 1) / blockSize.x;
	gridSize.x = (gridSize.x + 15) / 16;
	SquareSumKrnlUnrollLdg<16> << <gridSize.x, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
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
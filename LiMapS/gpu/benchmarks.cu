#include "benchmarks.cuh"

#include <iostream>

#include "cuda_shared.h"
#include "cublas_shared.h"

#include "kernels/misc.cuh"
#include "kernels/reduction.cuh"
#include "kernels/square_sum.cuh"
#include "kernels/matrix2vector.cuh"
#include "kernels/threshold.cuh"

__global__ void Fill(float* data, size_t size, float val = 1.0f) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		data[index] = val;
	}
}

void RunCopyBenchmarks(size_t dataSize) {
	cuda_ptr<float> source = make_cuda<float>(dataSize);
	cuda_ptr<float> dest = make_cuda<float>(dataSize);

	dim3 blockSize(128);
	CopyTo<8> << <GetGridSize(blockSize, dataSize, 8), blockSize >> > (source.get(), dataSize, dest.get(), false);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void RunThresholdBenchmarks(size_t dataSize) {
	cuda_ptr<float> data = make_cuda<float>(dataSize);

	dim3 blockSize(256);
	cudaMemset(data.get(), 0, dataSize * sizeof(float));

	std::cout << "Starting thresold kernel comparison benchmarks" << std::endl;

	blockSize.x = 32;
	ThresholdVector<1> << <GetGridSize(blockSize, dataSize), blockSize.x >> > (data.get(), dataSize);
	CUDA_CHECK(cudaDeviceSynchronize());

	ThresholdVectorAlwaysWrite<1> << <GetGridSize(blockSize, dataSize), blockSize.x >> > (data.get(), dataSize);
	CUDA_CHECK(cudaDeviceSynchronize());

	ThresholdVector<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x >> > (data.get(), dataSize);
	CUDA_CHECK(cudaDeviceSynchronize());

	ThresholdVectorAlwaysWrite<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x >> > (data.get(), dataSize);
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 128;
	ThresholdVector<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x >> > (data.get(), dataSize);
	CUDA_CHECK(cudaDeviceSynchronize());

	ThresholdVectorAlwaysWrite<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x >> > (data.get(), dataSize);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void RunNormBenchmarks(size_t dataSize) {
	cuda_ptr<float> data = make_cuda<float>(dataSize);

	dim3 blockSize(256);
	Fill << <GetGridSize(blockSize, dataSize), blockSize.x >> > (data.get(), dataSize);

	float norm = 0.0f;
	std::cout << "Starting NORM kernel comparison benchmarks" << std::endl;
	cublasHandle_t cublasHandle;
	CUBLAS_CHECK(cublasCreate(&cublasHandle));
	cublasSnrm2(cublasHandle, dataSize, data.get(), 1, &norm);
	CUBLAS_CHECK(cublasDestroy(cublasHandle));

	std::cout << "Cublas norm: " << norm << std::endl;

	cuda_ptr<float> deviceNorm = make_cuda<float>(1);

	blockSize.x = 32;
	SquareSumKrnl << <GetGridSize(blockSize, dataSize), blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel: " << norm << std::endl;

	blockSize.x = 64;
	SquareSumKrnl << <GetGridSize(blockSize, dataSize), blockSize, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 128;
	SquareSumKrnl << <GetGridSize(blockSize, dataSize), blockSize, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 256;
	SquareSumKrnl << <GetGridSize(blockSize, dataSize), blockSize, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());


	// 2-reduction kernels
	cudaMemset(deviceNorm.get(), 0, sizeof(float));

	blockSize.x = 32;
	SquareSumKrnlUnroll<2> << <GetGridSize(blockSize, dataSize, 2), blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel 2: " << norm << std::endl;

	blockSize.x = 64;
	SquareSumKrnlUnroll<2> << <GetGridSize(blockSize, dataSize, 2), blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 128;
	SquareSumKrnlUnroll<2> << <GetGridSize(blockSize, dataSize, 2), blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 256;
	SquareSumKrnlUnroll<2> << <GetGridSize(blockSize, dataSize, 2), blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	// 8-reduction kernels
	cudaMemset(deviceNorm.get(), 0, sizeof(float));

	blockSize.x = 64;
	SquareSumKrnlUnroll<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel 8: " << norm << std::endl;

	blockSize.x = 128;
	SquareSumKrnlUnroll<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 256;
	SquareSumKrnlUnroll<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	// 16-reduction with cache kernels
	cudaMemset(deviceNorm.get(), 0, sizeof(float));

	blockSize.x = 128;
	SquareSumGridUnroll<8> << <80, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "Norm from kernel 8 - ldg: " << norm << std::endl;

	blockSize.x = 256;
	SquareSumGridUnroll<8> << <80, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 512;
	SquareSumGridUnroll<8> << <80, blockSize.x, blockSize.x / 32 >> > (data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());
}

void RunNormDiffBenchmarks(size_t dataSize) {
	cuda_ptr<float> data2 = make_cuda<float>(dataSize);
	cuda_ptr<float> data = make_cuda<float>(dataSize);

	dim3 blockSize(256);
	dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
	Fill << <gridSize, blockSize.x >> > (data.get(), dataSize);
	Fill << <gridSize, blockSize.x >> > (data2.get(), dataSize, 2.0f);

	cuda_ptr<float> deviceNorm = make_cuda<float>(1);
	cudaMemset(deviceNorm.get(), 0, sizeof(float));

	// 8-reduction kernels
	blockSize.x = 64;
	SquareDiffSumKrnlUnroll<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data2.get(), data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	float norm = 0.0f;
	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "NormDiff from kernel 8: " << norm << std::endl;

	blockSize.x = 128;
	SquareDiffSumKrnlUnroll<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data2.get(), data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 256;
	SquareDiffSumKrnlUnroll<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data2.get(), data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	// Cache streaming kernels
	cudaMemset(deviceNorm.get(), 0, sizeof(float));

	blockSize.x = 128;
	SquareDiffSumKrnlUnrollLdg<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data2.get(), data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	cudaMemcpy(&norm, deviceNorm.get(), sizeof(float), cudaMemcpyDeviceToHost);
	norm = sqrt(norm);
	std::cout << "NormDiff from kernel 8 - ldg: " << norm << std::endl;

	blockSize.x = 256;
	SquareDiffSumKrnlUnrollLdg<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data2.get(), data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());

	blockSize.x = 512;
	SquareDiffSumKrnlUnrollLdg<8> << <GetGridSize(blockSize, dataSize, 8), blockSize.x, blockSize.x / 32 >> > (data2.get(), data.get(), dataSize, deviceNorm.get());
	CUDA_CHECK(cudaDeviceSynchronize());
}


void RunKernelsBenchmarks() {
	std::cout << "Starting benchmarks" << std::endl;

	size_t dataSize = 80000;
	cuda_ptr<float> zeroArray = make_cuda<float>(dataSize);
	cuda_ptr<float> destArray = make_cuda<float>(dataSize);

	dim3 blockSize(32);

}

void RunMatrixVectorBenchmarks(size_t dataSize) {

	size_t width = dataSize;
	size_t height = dataSize;
	std::cout << "Starting matrix -vector benchmarks" << std::endl;

	dim3 test = GetGridSize(dim3(128), width);

	cuda_ptr<float> matrix = make_cuda<float>(width * height);
	cuda_ptr<float> sourceArray = make_cuda<float>(test.x * 128);
	cuda_ptr<float> destArray = make_cuda<float>(width);

	dim3 blockSize(256);
	Fill << <GetGridSize(blockSize, width * height), blockSize >> > (matrix.get(), width * height);
	Fill << <GetGridSize(blockSize, width), blockSize >> > (sourceArray.get(), width);

#if NDEBUG
	/*std::cout << "Cublas ..." << std::endl;
	cublasHandle_t cublasHandle;
	CUBLAS_CHECK(cublasCreate(&cublasHandle));
	float alpha = 1.0f;
	CUBLAS_CHECK(cublasSgemv_v2(cublasHandle, CUBLAS_OP_T, height, width, &alpha, matrix.get(), height, sourceArray.get(), 1, &alpha, destArray.get(), 1));
	CUBLAS_CHECK(cublasDestroy(cublasHandle));*/
#endif

	dim3 gridSize;

	/*std::cout << "64 block size ..." << std::endl;
	blockSize.x = 64;
	gridSize = GetGridSize(blockSize, width, 8);
	gridSize.y = height;
	Matrix2Vector<8, false> << <gridSize, blockSize.x, blockSize.x / 32 >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize()); */

	std::cout << "128 block size ..." << std::endl;
	blockSize.x = 128;
	gridSize = GetGridSize(blockSize, width, 8);
	gridSize.y = height;
	Matrix2Vector<8, false> << <gridSize, blockSize.x, blockSize.x / 32 >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	/* std::cout << "256 block size ..." << std::endl;
	blockSize.x = 256;
	gridSize = GetGridSize(blockSize, width, 8);
	gridSize.y = height;
	Matrix2Vector<8, false> << <gridSize, blockSize.x, blockSize.x / 32 >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	std::cout << "512 block size ..." << std::endl;
	blockSize.x = 512;
	gridSize = GetGridSize(blockSize, width, 8);
	gridSize.y = height;
	Matrix2Vector<8, false> << <gridSize, blockSize.x, blockSize.x / 32 >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());*/


	/* Version 2*/

	/*std::cout << "v2 - 64 block size ..." << std::endl;
	blockSize.x = 64;
	gridSize = GetGridSize(blockSize, height);
	Matrix2Vector2B << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());*/

	std::cout << "v2 - 128 block size ..." << std::endl;
	blockSize.x = 128;
	gridSize = GetGridSize(blockSize, height);
	Matrix2Vector2 << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height,
		[] __device__(int dummy, float* dest, size_t idx, float sum) { dest[idx] = sum; }, 0);
	CUDA_CHECK(cudaDeviceSynchronize());


	std::cout << "v3 - 128 block size ..." << std::endl;
	blockSize.x = 128;
	gridSize = GetGridSize(blockSize, height);
	Matrix2VectorSharedMem << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	/*std::cout << "v2 - 256 block size ..." << std::endl;
	blockSize.x = 256;
	gridSize = GetGridSize(blockSize, height);
	Matrix2Vector2B << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	std::cout << "v2 - 512 block size ..." << std::endl;
	blockSize.x = 512;
	gridSize = GetGridSize(blockSize, height);
	Matrix2Vector2B << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());*/

	/* Version Partition Camping */

	/*std::cout << "vPart - 64 block size ..." << std::endl;
	blockSize.x = 64;
	gridSize = GetGridSize(blockSize, height);
	Matrix2VectorPartitionB << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());*/

	std::cout << "vPart - 128 block size ..." << std::endl;
	blockSize.x = 128;
	gridSize = GetGridSize(blockSize, height);
	Matrix2VectorPartitionB << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	/*std::cout << "vPart - 256 block size ..." << std::endl;
	blockSize.x = 256;
	gridSize = GetGridSize(blockSize, height);
	Matrix2VectorPartitionB << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

	std::cout << "vPart - 512 block size ..." << std::endl;
	blockSize.x = 512;
	gridSize = GetGridSize(blockSize, height);
	Matrix2VectorPartitionB << <gridSize, blockSize.x >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height);
	CUDA_CHECK(cudaDeviceSynchronize());*/

	/* Version 2 - stream */
	const int NSTREAM = 16;
	cudaStream_t cudaStreams[NSTREAM];
	for (int i = 0; i < NSTREAM; i++)
	{
		CUDA_CHECK(cudaStreamCreate(&cudaStreams[i]));
	}

	std::cout << "vStream - 128 block size ..." << std::endl;
	blockSize.x = 128;
	gridSize = GetGridSize(blockSize, (height + NSTREAM - 1) / NSTREAM);
	for (int i = 0; i < NSTREAM; i++)
	{
		size_t offset = ((height + NSTREAM - 1) / NSTREAM) * i;
		Matrix2VectorStream << <gridSize, blockSize.x, 0, cudaStreams[i] >> > (matrix.get(), sourceArray.get(), destArray.get(), width, height, offset);
	}
	CUDA_CHECK(cudaDeviceSynchronize());



	for (size_t i = 0; i < NSTREAM; i++)
	{
		CUDA_CHECK(cudaStreamDestroy(cudaStreams[i]));
	}

}
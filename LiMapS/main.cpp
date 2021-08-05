#include<iostream>
#include <vector>
#include <random>
#include <assert.h>
#include <fstream>
#include <string>
#include <sstream>

#include "StopWatch.h"

#include "gpu/benchmarks.cuh"

#include "cpu/HostLiMapS.h"
#include "gpu/DeviceLiMapSv1.cuh"
#include "gpu/DeviceLiMapSv2.cuh"
#include "gpu/DeviceLiMapSv3.cuh"
#include "gpu/DeviceLiMapSv4.cuh"
#include "gpu/DeviceLiMapSTex.cuh"

void RunLiMapS();

void ReadColumnVector(const std::string& file, float* dest) {
	std::ifstream stream(file);
	assert(stream.is_open());

	// Actual solution is a column vector, in the matrix file each element is on a single line
	int i = 0;
	float data;
	while (stream >> data) {
		dest[i++] = data;
	}
}

void ReadMatrix(const std::string& file, float* dest, size_t rows, size_t cols) {
	std::ifstream stream(file);
	assert(stream.is_open());

	std::string line;
	size_t rowIndex = 0;

	while (std::getline(stream, line))
	{
		std::istringstream lineStream(line);
		std::string floatData;
		size_t colIndex = 0;
		while (std::getline(lineStream, floatData, ','))
		{
			dest[rowIndex * cols + colIndex] = std::stof(floatData);
			++colIndex;
		}
		assert(colIndex == cols);

		++rowIndex;
	}
	assert(rowIndex == rows);
}

void RunLiMapSOnCPU(const float* dictionary, const float* dictionaryInverse, const float* signal, const float* actualSolution, size_t dictionaryWords, size_t signalSize, int maxIterations) {
	std::cout << "Starting CPU execution ..." << std::endl;

	HostLiMapS hostLiMapS(actualSolution, signal, dictionary, dictionaryInverse, dictionaryWords, signalSize);

	StopWatch sw;
	sw.Restart();
	hostLiMapS.Execute(maxIterations);
	sw.Stop();

	const std::vector<float>& alphaResult = hostLiMapS.GetAlpha();
	for (size_t i = 0; i < dictionaryWords; i++)
	{
		if (actualSolution[i] != alphaResult[i])
		{
			std::cout << "Actual solution[" << i << "] mismatch: " << actualSolution[i] << " from solution, " << alphaResult[i] << " from cpu" << std::endl;
		}
	}
	std::cout << "CPU execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;
}

void RunLiMapSOnCuBlas(const float* dictionary, const float* dictionaryInverse, const float* signal, const float* actualSolution, size_t dictionaryWords, size_t signalSize, int maxIterations) {
	std::cout << "Starting CuBlas  execution ..." << std::endl;

	DeviceLiMapSv1 deviceLiMapSV1(actualSolution, signal, dictionary, dictionaryInverse, dictionaryWords, signalSize);

	StopWatch sw;
	sw.Restart();
	deviceLiMapSV1.Execute(maxIterations);
	sw.Stop();

	const std::vector<float>& alphaResult = deviceLiMapSV1.GetAlpha();
	for (size_t i = 0; i < dictionaryWords; i++)
	{
		if (actualSolution[i] != alphaResult[i])
		{
			std::cout << "Actual solution[" << i << "] mismatch: " << actualSolution[i] << " from solution, " << alphaResult[i] << " from CuBlas" << std::endl;
		}
	}

	std::cout << "CuBlas execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;
}

void RunBaseLiMapSKernel(const float* dictionary, const float* dictionaryInverse, const float* signal, const float* actualSolution, size_t dictionaryWords, size_t signalSize, int maxIterations) {
	std::cout << "Starting GPU kernel (naive) execution ..." << std::endl;
	DeviceLiMapSv2 deviceLiMapSv2(actualSolution, signal, dictionary, dictionaryInverse, dictionaryWords, signalSize);

	StopWatch sw;
	sw.Restart();
	deviceLiMapSv2.Execute(maxIterations);
	sw.Stop();

	const std::vector<float>& alphaResult = deviceLiMapSv2.GetAlpha();
	for (size_t i = 0; i < dictionaryWords; i++)
	{
		if (actualSolution[i] != alphaResult[i])
		{
			std::cout << "Actual solution[" << i << "] mismatch: " << actualSolution[i] << " from solution, " << alphaResult[i] << " from GPU" << std::endl;
		}
	}
	std::cout << "GPU kernel (naive) execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;
}

void RunImprovedLiMapSKernel(const float* dictionary, const float* dictionaryInverse, const float* signal, const float* actualSolution, size_t dictionaryWords, size_t signalSize, int maxIterations) {
	std::cout << "Starting GPU kernel (final) execution ..." << std::endl;
	DeviceLiMapSv3 deviceLiMapSv3(actualSolution, signal, dictionary, dictionaryInverse, dictionaryWords, signalSize);

	StopWatch sw;
	sw.Restart();
	deviceLiMapSv3.Execute(maxIterations);
	sw.Stop();

	const std::vector<float>& alphaResult = deviceLiMapSv3.GetAlpha();
	for (size_t i = 0; i < dictionaryWords; i++)
	{
		if (actualSolution[i] != alphaResult[i])
		{
			std::cout << "Actual solution[" << i << "] mismatch: " << actualSolution[i] << " from solution, " << alphaResult[i] << " from GPU" << std::endl;
		}
	}
	std::cout << "GPU kernel (final) execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;
}

void RunFinalLiMapSKernel(const float* dictionary, const float* dictionaryInverse, const float* signal, const float* actualSolution, size_t dictionaryWords, size_t signalSize, int maxIterations) {
	std::cout << "Starting GPU kernel (final) execution ..." << std::endl;
	DeviceLiMapSv4 deviceLiMapSv4(actualSolution, signal, dictionary, dictionaryInverse, dictionaryWords, signalSize);

	StopWatch sw;
	sw.Restart();
	deviceLiMapSv4.Execute(maxIterations);
	sw.Stop();

	const std::vector<float>& alphaResult = deviceLiMapSv4.GetAlpha();
	for (size_t i = 0; i < dictionaryWords; i++)
	{
		if (actualSolution[i] != alphaResult[i])
		{
			std::cout << "Actual solution[" << i << "] mismatch: " << actualSolution[i] << " from solution, " << alphaResult[i] << " from GPU" << std::endl;
		}
	}
	std::cout << "GPU kernel (final) execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;
}

void RunTexLiMapSKernel(const float* dictionary, const float* dictionaryInverse, const float* signal, const float* actualSolution, size_t dictionaryWords, size_t signalSize, int maxIterations) {
	std::cout << "Starting GPU kernel (texture) execution ..." << std::endl;
	DeviceLiMapSTex deviceLiMapSTex(actualSolution, signal, dictionary, dictionaryInverse, dictionaryWords, signalSize);

	StopWatch sw;
	sw.Restart();
	deviceLiMapSTex.Execute(maxIterations);
	sw.Stop();

	const std::vector<float>& alphaResult = deviceLiMapSTex.GetAlpha();
	for (size_t i = 0; i < dictionaryWords; i++)
	{
		if (actualSolution[i] != alphaResult[i])
		{
			std::cout << "Actual solution[" << i << "] mismatch: " << actualSolution[i] << " from solution, " << alphaResult[i] << " from GPU" << std::endl;
		}
	}
	std::cout << "GPU kernel (texture) execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;
}

int main(int argn, char** argc)
{
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	std::cout << props.name << std::endl;
	std::cout << "\tThreads per block: " << props.maxThreadsPerBlock << std::endl;

	if (argn > 1 && strcmp(argc[1], "norm-benchmark") == 0) {
		RunNormBenchmarks(atoi(argc[2]));
	}
	else if (argn > 1 && strcmp(argc[1], "normdiff-benchmark") == 0) {
		RunNormDiffBenchmarks(atoi(argc[2]));
	}
	else if (argn > 1 && strcmp(argc[1], "th-benchmark") == 0) {
		RunThresholdBenchmarks(atoi(argc[2]));
	}
	else if (argn > 1 && strcmp(argc[1], "copy-benchmark") == 0) {
		RunCopyBenchmarks(atoi(argc[2]));
	}
	else if (argn > 1 && strcmp(argc[1], "mv-benchmark") == 0) {
		RunMatrixVectorBenchmarks(atoi(argc[2]));
	}
	else {
		RunLiMapS();
	}	
	return 0;
}

void RunLiMapS() {
	// Let' s  change the floating point precision when printing
	std::cout.precision(std::numeric_limits<float>::max_digits10);
	std::cout << " *** LiMapS Implementation ***" << std::endl;

	const int signalSize = 2000;
	const int dictionaryWords = 8000;

	// We may use some async CUDA memories operation so better to declare our pointer as non-paginable memory
	float* actualSolution;
	float* signal;
	float* dictionary;
	float* dictionaryInverse;

	CUDA_CHECK(cudaMallocHost(&actualSolution, sizeof(float) * dictionaryWords));
	CUDA_CHECK(cudaMallocHost(&signal, sizeof(float) * signalSize));
	CUDA_CHECK(cudaMallocHost(&dictionary, sizeof(float) * dictionaryWords * signalSize));
	CUDA_CHECK(cudaMallocHost(&dictionaryInverse, sizeof(float) * dictionaryWords * signalSize));

	std::cout << "Reading data ..." << std::endl;

	// Let' s read our data from a file for the moment and assert that evertything has the right dimension
	ReadColumnVector("data\\2\\in_true_alpha.txt", actualSolution);
	ReadColumnVector("data\\2\\in_signal.txt", signal);
	ReadMatrix("data\\2\\in_D.txt", dictionary, signalSize, dictionaryWords);
	ReadMatrix("data\\2\\in_D_inverse.txt", dictionaryInverse, dictionaryWords, signalSize);

		std::cout << "# Dictionary atoms: " << dictionaryWords << std::endl;
	std::cout << "Signal size: " << signalSize << std::endl << std::endl;

	// Stopping criteria declaration
	const int maxIterations = 1000;

	RunLiMapSOnCPU(dictionary, dictionaryInverse, signal, actualSolution, dictionaryWords, signalSize, maxIterations);

#if NDEBUG
	// Cublas makes NSight debugging hang, so better to not use it
	// It will be used only in release mode, as comparison 
	RunLiMapSOnCuBlas(dictionary, dictionaryInverse, signal, actualSolution, dictionaryWords, signalSize, maxIterations);
#endif

	//RunBaseLiMapSKernel(dictionary, dictionaryInverse, signal, actualSolution, dictionaryWords, signalSize, maxIterations);
	//RunImprovedLiMapSKernel(dictionary, dictionaryInverse, signal, actualSolution, dictionaryWords, signalSize, maxIterations);
	RunFinalLiMapSKernel(dictionary, dictionaryInverse, signal, actualSolution, dictionaryWords, signalSize, maxIterations);
	//RunTexLiMapSKernel(dictionary, dictionaryInverse, signal, actualSolution, dictionaryWords, signalSize, maxIterations);

	/* ----- Cleanup ----- */

	CUDA_CHECK(cudaFreeHost(actualSolution));
	CUDA_CHECK(cudaFreeHost(signal));
	CUDA_CHECK(cudaFreeHost(dictionary));
	CUDA_CHECK(cudaFreeHost(dictionaryInverse));
}

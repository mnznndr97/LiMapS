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

template <class T>
void ReadColumnVector(const std::string& file, std::vector<T>& dest) {
	std::ifstream stream(file);
	assert(stream.is_open());

	// Actual solution is a column vector, in the matrix file each element is on a single line
	T data;
	while (stream >> data) {
		dest.push_back(data);
	}
}

void ReadMatrix(const std::string& file, std::vector<float>& dest, size_t rows, size_t cols) {
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

int main(int argn, char** argc)
{
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	std::cout << props.name << std::endl;
	std::cout <<  "\tThreads per block: " << props.maxThreadsPerBlock << std::endl;

	if (argn > 1 && strcmp(argc[1], "norm-benchmark") == 0) {
		RunNormBenchmarks(atoi(argc[2]));
		
		return 0;
	}

	std::cout << " *** LiMapS Implementation ***" << std::endl;

	const int signalSize = 200;
	const int dictionaryWords = 800;

	std::vector<float> actualSolution;
	std::vector<float> signal;
	std::vector<float> dictionary(signalSize * dictionaryWords);
	std::vector<float> dictionaryInverse(signalSize * dictionaryWords);

	// Let' s read our data from a file for the moment and assert that evertything has the right dimension
	ReadColumnVector("data\\1\\in_true_alpha.txt", actualSolution);
	ReadColumnVector("data\\1\\in_signal.txt", signal);
	ReadMatrix("data\\1\\in_D.txt", dictionary, signalSize, dictionaryWords);
	ReadMatrix("data\\1\\in_D_inverse.txt", dictionaryInverse, dictionaryWords, signalSize);

	assert(actualSolution.size() == dictionaryWords);
	assert(signal.size() == signalSize);

	std::cout << "# Dictionary atoms: " << dictionaryWords << std::endl;
	std::cout << "Signal size: " << signalSize << std::endl << std::endl;

	StopWatch sw;

	// Stopping criteria declaration
	const float epsilon = 1e-5;
	const int maxIterations = 1000;

	std::cout << "Starting CPU execution ..." << std::endl;
	{
		// Just enclose the call in an inner scope to release as soon as possible the extra host resources
		HostLiMapS hostLiMapS(actualSolution, signal, dictionary, dictionaryInverse);

		sw.Restart();
		hostLiMapS.Execute(maxIterations);
		sw.Stop();
	}
	std::cout << "CPU execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;

	std::cout << "Starting CuBlas (naive) execution ..." << std::endl;
	{
		// Just enclose the call in an inner scope to release as soon as possible the GPU resources
#if NDEBUG
		/*DeviceLiMapSv1 deviceLiMapSV1(actualSolution, signal, dictionary, dictionaryInverse);

		sw.Restart();
		deviceLiMapSV1.Execute(maxIterations);
		sw.Stop();*/
#endif
	}
	std::cout << "CuBlas (naive) execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;

	std::cout << "Starting GPU kernel execution ..." << std::endl;
	{
		// Just enclose the call in an inner scope to release as soon as possible the GPU resources
		DeviceLiMapSv2 deviceLiMapSv2(actualSolution, signal, dictionary, dictionaryInverse);

		sw.Restart();
		deviceLiMapSv2.Execute(maxIterations);
		sw.Stop();
	}
	std::cout << "GPU kernel  execution time: " << sw.Elapsed() << " ms" << std::endl << std::endl;

	return 0;
}

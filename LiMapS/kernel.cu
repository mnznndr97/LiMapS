#include<iostream>
#include <vector>
#include <random>
#include <assert.h>
#include <fstream>
#include <string>
#include <sstream>

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

template <class T>
void ReadMatrix(const std::string& file, std::unique_ptr<T[]>& dest, int rows, int cols) {
	std::ifstream stream(file);
	assert(stream.is_open());

	std::string line;
	T data;
	int rowIndex = 0;

	while (std::getline(stream, line))
	{
		std::istringstream lineStream(line);
		std::string floatData;
		int colIndex = 0;
		while (std::getline(lineStream, floatData, ','))
		{
			++colIndex;
		}
		assert(colIndex == cols);

		++rowIndex;
	}
	assert(rowIndex == rows);
	
}

int main()
{
	std::cout << " *** LiMapS Implementation ***" << std::endl;

	const int signalSize = 200;
	const int dictionaryWords = 800;

	std::vector<float> actualSolution;
	std::vector<float> signal;
	std::unique_ptr<float[]> D = std::make_unique<float[]>(signalSize * dictionaryWords);

	// Let' s read our data from a file for the moment and assert that evertything has the right dimension
	ReadColumnVector("data\\1\\in_true_alpha.txt", actualSolution);
	ReadColumnVector("data\\1\\in_signal.txt", signal);
	ReadMatrix("data\\1\\in_D.txt", D, signalSize, dictionaryWords);

	assert(actualSolution.size() == dictionaryWords);
	assert(signal.size() == signalSize);

	std::cout << "# Dictionary atoms: " << dictionaryWords << std::endl;
	std::cout << "Signal size: " << signalSize << std::endl;

	return 0;
}


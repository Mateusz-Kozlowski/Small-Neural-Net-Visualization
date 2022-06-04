#pragma once

#include "NeuralNet.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>

namespace Utils
{
	namespace MNISTdataLoader
	{
		unsigned char readChar(std::ifstream& file);

		int readInt(std::ifstream& file);

		std::vector<std::vector<Scalar>> readMnistImages(const char* filename);

		std::vector<int> readMnistLabels(const char* filename);

		void showImage(const std::vector<Scalar>& image);

		void showData(
			const std::vector<std::vector<Scalar>>& images,
			const std::vector<int>& labels,
			std::pair<unsigned, unsigned> index_range
		);

		void preprocess(std::vector<std::vector<Scalar>>& images);

		std::vector<std::vector<Scalar>> reorganizeLabels(const std::vector<int>& labels);
	}
}

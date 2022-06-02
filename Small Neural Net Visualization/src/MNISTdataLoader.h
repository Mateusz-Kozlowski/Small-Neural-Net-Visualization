#pragma once

#include "NeuralNet.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>

unsigned char readChar(std::ifstream& file)
{
	char c;
	file.read(&c, 1);
	return c;
}

int readInt(std::ifstream& file)
{
	int result = 0;

	for (int byte = 0; byte < 4; byte++) result = (result <<= 8) + readChar(file);

	return result;
}

std::vector<std::vector<Scalar>> readMnistImages(const char* filename)
{
	std::ifstream file(filename, std::ios::binary);

	if (!file.good())
	{
		std::cerr << "CANNOT OPEN: " << filename << '\n';
	}

	// first number in the file is a checksum
	const int CHECKSUM_EXPECTED = 2051;
	int checksum = readInt(file);
	assert(checksum == CHECKSUM_EXPECTED);

	int imagesCount = readInt(file);
	int rowsCount = readInt(file);
	int columnsCount = readInt(file);
	int imageSize = rowsCount * columnsCount;

	std::vector<std::vector<Scalar>> images(imagesCount, std::vector<Scalar>(imageSize));

	for (auto& image : images) for (Scalar& pixel : image) pixel = static_cast<Scalar>(readChar(file));

	file.close();

	return images;
}

std::vector<int> readMnistLabels(const char* filename)
{
	std::ifstream file(filename, std::ios::binary);

	if (!file.good())
	{
		std::cerr << "CANNOT OPEN: " << filename << '\n';
	}

	// first number in the file is a checksum
	const int CHECKSUM_EXPECTED = 2049;
	int checksum = readInt(file);
	assert(checksum == CHECKSUM_EXPECTED);

	int number_of_labels = readInt(file);

	std::vector<int> labels(number_of_labels);

	for (int& label : labels) label = readChar(file);

	file.close();

	return labels;
}

void showImage(const std::vector<Scalar>& image)
{
	for (int i = 0; i < image.size(); i++)
	{
		if (image[i] > 0.5) std::cout << '#';

		else std::cout << '.';

		if (i % 28 == 27) std::cout << '\n';
	}
}

void showData(
	const std::vector<std::vector<Scalar>>& images,
	const std::vector<int>& labels,
	std::pair<unsigned, unsigned> index_range)
{
	for (int i = index_range.first; i <= index_range.second; i++)
	{
		std::cout << labels[i] << ":\n";
		showImage(images[i]);
		std::cout << '\n';
	}
}

void preprocess(std::vector<std::vector<Scalar>>& images)
{
	for (auto& image : images)
	{
		for (auto& pixel : image)
		{
			pixel /= 255.0;
		}
	}
}

std::vector<std::vector<Scalar>> reorganizeLabels(const std::vector<int>& labels)
{
	std::vector<std::vector<Scalar>> reorganizedLabels;

	reorganizedLabels.reserve(labels.size());

	std::vector<Scalar> temp(10, 0.0);

	for (auto& label : labels)
	{
		temp[label] = 1.0;
		reorganizedLabels.push_back(temp);
		temp[label] = 0.0;
	}

	return reorganizedLabels;
}

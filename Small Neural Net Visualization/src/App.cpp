#include "App.h"

void App::run()
{
	std::cout << "Loading data...\n";

	auto trainInputs = Utils::MNISTdataLoader::readMnistImages(
		"handwritten digits data//train-images-idx3-ubyte"
	);
	auto trainLabels = Utils::MNISTdataLoader::readMnistLabels(
		"handwritten digits data//train-labels-idx1-ubyte"
	);

	auto testInputs = Utils::MNISTdataLoader::readMnistImages(
		"handwritten digits data//t10k-images-idx3-ubyte"
	);
	auto testLabels = Utils::MNISTdataLoader::readMnistLabels(
		"handwritten digits data//t10k-labels-idx1-ubyte"
	);

	std::cout << "\nThis is how 2 first data points look like:\n";

	Utils::MNISTdataLoader::showData(trainInputs, trainLabels, { 0U, 1U });

	Utils::MNISTdataLoader::preprocess(trainInputs);
	Utils::MNISTdataLoader::preprocess(testInputs);

	auto reorganisedTrainLabels = Utils::MNISTdataLoader::reorganizeLabels(trainLabels);
	auto reorganisedTestLabels = Utils::MNISTdataLoader::reorganizeLabels(testLabels);

	NeuralNet net({ 784U, 16U, 12U, 10U }, 1.0, 32U);

	unsigned epochsCount = 1U;
	std::cout << "Learning epochs count: ";
	std::cin >> epochsCount;

	for (int e = 1; e <= epochsCount; e++)
	{
		for (int i = 0; i < trainInputs.size(); i++)
		{
			net.trainingStep(trainInputs[i], reorganisedTrainLabels[i]);
		}

		std::cout << "Accuracy after " << e << " epoch: " << Utils::validateClassification(
			testInputs,
			reorganisedTestLabels,
			net
		) << '\n';

		Utils::randomShuffle(trainInputs, reorganisedTrainLabels);
	}
}

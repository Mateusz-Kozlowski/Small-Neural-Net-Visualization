#include "MNISTdataLoader.h"

Scalar validateClassification(
	const std::vector<std::vector<Scalar>>& validationDataInputs,
	const std::vector<std::vector<Scalar>>& validationDataLabels,
	NeuralNet& net
);

void randomShuffle(
	std::vector<std::vector<Scalar>>& inputs,
	std::vector<std::vector<Scalar>>& desiredOutputs
);

int main()
{
	std::cout << "Loading data...";
	
	auto trainInputs = MNISTdataLoader::readMnistImages(
		"handwritten digits data//train-images-idx3-ubyte"
	);
	auto trainLabels = MNISTdataLoader::readMnistLabels(
		"handwritten digits data//train-labels-idx1-ubyte"
	);

	auto testInputs = MNISTdataLoader::readMnistImages(
		"handwritten digits data//t10k-images-idx3-ubyte"
	);
	auto testLabels = MNISTdataLoader::readMnistLabels(
		"handwritten digits data//t10k-labels-idx1-ubyte"
	);

	std::cout << "\n\n";

	MNISTdataLoader::showData(trainInputs, trainLabels, { 0U, 1U });

	MNISTdataLoader::preprocess(trainInputs);
	MNISTdataLoader::preprocess(testInputs);

	auto reorganisedTrainLabels = MNISTdataLoader::reorganizeLabels(trainLabels);
	auto reorganisedTestLabels = MNISTdataLoader::reorganizeLabels(testLabels);

	NeuralNet net({ 784U, 32U, 16U, 10U }, 1.0, 32U);

	unsigned epochsCount = 1U;
	std::cout << "Epochs count: ";
	std::cin >> epochsCount;

	for (int e = 1; e <= epochsCount; e++)
	{
		for (int i = 0; i < trainInputs.size(); i++)
		{
			net.trainingStep(trainInputs[i], reorganisedTrainLabels[i]);
		}

		std::cout << "Accuracy after " << e << " epochs: " << validateClassification(
			testInputs,
			reorganisedTestLabels,
			net
		) << '\n';

		randomShuffle(trainInputs, reorganisedTrainLabels);
	}
}

Scalar validateClassification(
	const std::vector<std::vector<Scalar>>& validationDataInputs,
	const std::vector<std::vector<Scalar>>& validationDataLabels,
	NeuralNet& net)
{
	unsigned goodAnswers = 0U;

	for (int i = 0; i < validationDataInputs.size(); i++)
	{
		std::vector<Scalar> predictions = net.predict(validationDataInputs[i]);

		// find max index in predictions:
		unsigned maxIndex1 = 0U;

		for (int j = 1; j < predictions.size(); j++) 
		{
			if (predictions[j] > predictions[maxIndex1])
			{
				maxIndex1 = j;
			}
		}

		// find max index in validation outputs:
		unsigned maxIndex2 = 0U;

		for (int j = 1; j < predictions.size(); j++)
		{
			if (validationDataLabels[i][j] > validationDataLabels[i][maxIndex2])
			{
				maxIndex2 = j;
			}
		}

		goodAnswers += maxIndex1 == maxIndex2;
	}

	return static_cast<Scalar>(goodAnswers) / static_cast<Scalar>(validationDataInputs.size());
}

void randomShuffle(
	std::vector<std::vector<Scalar>>& inputs,
	std::vector<std::vector<Scalar>>& desiredOutputs)
{
	for (int i = 0; i < inputs.size(); i++)
	{
		unsigned randomIndex = RandomEngine::getIntInRange(0U, inputs.size() - 1);

		std::swap(inputs[randomIndex], inputs[i]);
		std::swap(desiredOutputs[randomIndex], desiredOutputs[i]);
	}
}

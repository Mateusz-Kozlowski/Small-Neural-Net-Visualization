#include "Utils.h"

Scalar Utils::validateClassification(
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

void Utils::randomShuffle(
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

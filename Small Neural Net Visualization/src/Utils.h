#pragma once

#include "MNISTdataLoader.h"

namespace Utils
{
	Scalar validateClassification(
		const std::vector<std::vector<Scalar>>& validationDataInputs,
		const std::vector<std::vector<Scalar>>& validationDataLabels,
		NeuralNet& net
	);

	void randomShuffle(
		std::vector<std::vector<Scalar>>& inputs,
		std::vector<std::vector<Scalar>>& desiredOutputs
	);
}

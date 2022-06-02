#pragma once

#include <vector>
#include <random>

#include "Config.h"

class NeuralNet
{
public:
	NeuralNet(
		const std::vector<unsigned>& topology,
		const Scalar& learningRate,
		unsigned miniBatchSize
	);

	const std::vector<Scalar>& predict(const std::vector<Scalar>& input);
	
	void trainingStep(
		const std::vector<Scalar>& input,
		const std::vector<Scalar>& desiredOutput
	);

private:
	void initVals(const std::vector<unsigned>& topology);
	void initWeights(const std::vector<unsigned>& topology);
	void initBiases(const std::vector<unsigned>& topology);

	const std::vector<Scalar>& getOutput() const;

	void propagateForward(const std::vector<Scalar>& input);

	void calcDerivatives();
	void calcErrors(const std::vector<Scalar>& desiredOutputs);
	void propagateErrorsBack();
	
	void updateGradients();
	void updateWeightsGradients();
	void updateBiasesGradients();

	void updateWeights();
	void updateBiases();

	void resetGradients();
	void resetWeightsGradients();
	void resetBiasesGradients();

	unsigned m_trainingStep;

	Scalar m_learningRate;
	unsigned m_miniBatchSize;
	
	std::vector<std::vector<Scalar>> m_val;
	std::vector<std::vector<Scalar>> m_biases;
	std::vector<std::vector<Scalar>> m_actVal;
	std::vector<std::vector<Scalar>> m_derivatives;
	std::vector<std::vector<Scalar>> m_lossDerivativeWithRespectToActFunc;

	std::vector<std::vector<std::vector<Scalar>>> m_weights;

	std::vector<std::vector<Scalar>> m_biasesGradient;
	std::vector<std::vector<std::vector<Scalar>>> m_weightsGradient;
};

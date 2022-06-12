#pragma once

#include "NeuralLayer.h"

class OutputLayer : public NeuralLayer
{
public:
	OutputLayer(
		unsigned size,
		const sf::Vector2f& pos,
		const sf::Color& bgColor,
		float neuronCircleDiameter,
		float distBetweenNeuronsCircles
	);

	virtual void propagateForward(
		const std::vector<Scalar>& inputVector,
		const SynapsesMatrix& inputSynapses
	) override;

	virtual const std::vector<Scalar>& getOutput() override;

	virtual void calcErrors(const std::vector<Scalar>& desiredOutputs) override;

	virtual void propagateErrorsBack(
		const Layer& nextLayer,
		const SynapsesMatrix& outputSynapses
	) override;

private:
	std::vector<Scalar> m_output;
};

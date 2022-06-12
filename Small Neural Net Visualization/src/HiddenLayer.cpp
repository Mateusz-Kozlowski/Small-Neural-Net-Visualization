#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(
	unsigned size,
	const sf::Vector2f& pos,
	const sf::Color& bgColor,
	float neuronCircleDiameter,
	float distBetweenNeuronsCircles)
{
	initNeurons(
		size, 
		pos, 
		neuronCircleDiameter,
		distBetweenNeuronsCircles
	);
	initBg(
		pos,
		bgColor,
		neuronCircleDiameter,
		distBetweenNeuronsCircles
	);
}

void HiddenLayer::propagateForward(
	const std::vector<Scalar>& inputVector,
	const SynapsesMatrix& inputSynapses)
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		Scalar input = 0.0;
		for (int p = 0; p < inputVector.size(); p++)
		{
			input += inputVector[p] * inputSynapses.getWeight(i, p);
		}

		m_neurons[i].setVal(input);
		m_neurons[i].activate();
	}
}

const std::vector<Scalar>& HiddenLayer::getOutput()
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void HiddenLayer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void HiddenLayer::propagateErrorsBack(
	const Layer& nextLayer,
	const SynapsesMatrix& outputSynapses)
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].calcLossDerivativeWithRespectToActFunc(
			i,
			nextLayer.getNeurons(),
			outputSynapses.getSynapsesMatrix()
		);
	}
}

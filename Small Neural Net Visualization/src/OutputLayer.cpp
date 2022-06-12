#include "OutputLayer.h"

OutputLayer::OutputLayer(
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
	m_output.resize(size);	
}

void OutputLayer::propagateForward(
	const std::vector<Scalar>& inputVector,
	const SynapsesMatrix& inputSynapses)
{
	std::cerr << "OutputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Scalar>& OutputLayer::getOutput()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_output[i] = m_neurons[i].getActVal();
	}

	return m_output;
}

void OutputLayer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	for (int i = 0; i < desiredOutputs.size(); i++)
	{
		m_neurons[i].calcLossDerivativeWithRespectToActFunc(desiredOutputs[i]);
	}
}

void OutputLayer::propagateErrorsBack(
	const Layer& nextLayer,
	const SynapsesMatrix& outputSynapses)
{
	std::cerr << "OutputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

#include "InputLayer.h"

InputLayer::InputLayer(unsigned size) : Layer(size)
{
	m_input.resize(size);
}

void InputLayer::setInput(const std::vector<Scalar>& input)
{
	m_input = input;
}

const std::vector<Scalar>& InputLayer::getInput() const
{
	return m_input;
}

void InputLayer::propagateForward(
	const std::vector<Scalar>& inputVector, 
	const SynapsesMatrix& inputSynapses)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::propagateForward(
	const Layer& previousLayer, 
	const SynapsesMatrix& inputSynapses)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Scalar>& InputLayer::getOutput()
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::calcDerivatives()
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::propagateErrorsBack(
	const Layer& nextLayer, 
	const SynapsesMatrix& outputSynapses)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Neuron>& InputLayer::getNeurons() const
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::updateBiasesGradients()
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

unsigned InputLayer::getSize() const
{
	return m_input.size();
}

void InputLayer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const Scalar& InputLayer::getBias(unsigned neuronIdx) const
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::resetBiasesGradients()
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

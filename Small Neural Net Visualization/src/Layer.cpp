#include "Layer.h"

Layer::Layer(unsigned size)
	: m_null(0.0)
{
	
}

const Scalar& Layer::getVal(unsigned idx) const
{
	std::cout << "NIEPOROZUMIENIE 1\n";
	exit(-17);
	return m_null;
}

const std::vector<Scalar>& Layer::getInput() const
{
	return m_nullVector;
}

void Layer::setInput(const std::vector<Scalar>& input)
{
	std::cout << "NIEPOROZUMIENIE 2 \n";
	exit(-17);
}

void Layer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	std::cout << "NIEPOROZUMIENIE3\n";
	exit(-17);
}

void Layer::propagateForward(
	const std::vector<Scalar>& input, 
	const SynapsesMatrix& inputSynapses)
{
	std::cout << "NIEPOROZUMIENIE0\n";
	exit(-17);
}

void Layer::propagateForward(
	const Layer& previousLayer, 
	const SynapsesMatrix& inputSynapses)
{
	std::cout << "NIEPOROZUMIENIE4\n";
	exit(-17);
}

void Layer::calcDerivatives()
{
	std::cout << "NIEPOROZUMIENIE5\n";
	exit(-17);
}

void Layer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	std::cout << "NIEPOROZUMIENIE6\n";
	exit(-17);
}

void Layer::calcErrors(
	const std::vector<Neuron>& nextLayerNeurons, 
	const std::vector<std::vector<Scalar>>& outputSynapsesWeights)
{
	std::cout << "NIEPOROZUMIENIE7\n";
	exit(-17);
}

void Layer::propagateErrorsBack(
	const Layer& nextLayer, 
	const SynapsesMatrix& outputSynapses)
{
	std::cout << "NIEPOROZUMIENIE8\n";
	exit(-17);
}

const Scalar& Layer::getActVal(unsigned neuronIdx) const
{
	std::cout << "NIEPOROZUMIENIE9n";
	exit(-17);
	return m_null;
}

const Scalar& Layer::getBias(unsigned neuronIdx) const
{
	std::cout << "NIEPOROZUMIENIE10\n";
	exit(-17);
	return m_null;
}

const Scalar& Layer::getDerivative(unsigned neuronIdx) const
{
	std::cout << "NIEPOROZUMIENIE11\n";
	exit(-17);
	return m_null;
}

const Scalar& Layer::getLossDerivativeWithRespectToActFunc(unsigned neuronIdx) const
{
	std::cout << "NIEPOROZUMIENIE12\n";
	exit(-17);
	return m_null;
}

const std::vector<Neuron>& Layer::getNeurons() const
{
	std::cout << "NIEPOROZUMIENIE13\n";
	exit(-17);
	return m_nullNeuronsVector;
}

const std::vector<Scalar>& Layer::getOutput()
{
	std::cout << "NIEPOROZUMIENIE14\n";
	exit(-17);
	return m_nullVector;
}

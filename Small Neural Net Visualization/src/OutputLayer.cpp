#include "OutputLayer.h"

OutputLayer::OutputLayer(unsigned size) : Layer(size)
{
	for (int i = 0; i < size; i++)
	{
		m_neurons.emplace_back(Neuron());
	}

	m_output.resize(size);
}

unsigned OutputLayer::getSize() const
{
	return m_neurons.size();
}

void OutputLayer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	m_neurons[neuronIdx].setBias(bias);
}

void OutputLayer::propagateForward(
	const Layer& previousLayer, 
	const SynapsesMatrix& inputSynapses)
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		Scalar input = 0.0;
		for (int p = 0; p < previousLayer.getNeurons().size(); p++)
		{
			input += previousLayer.getNeurons()[p].getActVal() * inputSynapses.getWeight(i, p);
		}

		m_neurons[i].setVal(input);
		m_neurons[i].activate();
	}
}

void OutputLayer::calcDerivatives()
{
	for (auto& neuron : m_neurons)
	{
		neuron.calcDerivative();
	}
}

void OutputLayer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	for (int i=0; i<desiredOutputs.size(); i++)
	{
		m_neurons[i].calcLossDerivativeWithRespectToActFunc(desiredOutputs[i]);
	}
}

const Scalar& OutputLayer::getActVal(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getActVal();
}

const Scalar& OutputLayer::getBias(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getBias();
}

const Scalar& OutputLayer::getDerivative(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getDerivative();
}

const Scalar& OutputLayer::getLossDerivativeWithRespectToActFunc(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getLossDerivativeWithRespectToActFunc();
}

const std::vector<Scalar>& OutputLayer::getOutput()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_output[i] = m_neurons[i].getActVal();
	}

	return m_output;
}

const std::vector<Neuron>& OutputLayer::getNeurons() const
{
	return m_neurons;
}

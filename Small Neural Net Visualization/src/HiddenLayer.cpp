#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(unsigned size) : Layer(size)
{
	for (int i = 0; i < size; i++)
	{
		m_neurons.emplace_back(Neuron());
	}
}

void HiddenLayer::updateBiasesGradients()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].updateBiasGradient();
	}
}

unsigned HiddenLayer::getSize() const
{
	return m_neurons.size();
}

void HiddenLayer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	m_neurons[neuronIdx].setBias(bias);
}

void HiddenLayer::resetBiasesGradients()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].resetBiasGradient();
	}
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

void HiddenLayer::propagateForward(
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

void HiddenLayer::calcDerivatives()
{
	for (auto& neuron : m_neurons)
	{
		neuron.calcDerivative();
	}
}

const Scalar& HiddenLayer::getActVal(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getActVal();
}

const Scalar& HiddenLayer::getBias(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getBias();
}

const Scalar& HiddenLayer::getDerivative(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getDerivative();
}

const Scalar& HiddenLayer::getLossDerivativeWithRespectToActFunc(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getLossDerivativeWithRespectToActFunc();
}

const std::vector<Neuron>& HiddenLayer::getNeurons() const
{
	return m_neurons;
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

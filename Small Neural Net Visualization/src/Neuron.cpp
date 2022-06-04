#include "Neuron.h"

void Neuron::setVal(const Scalar& val)
{
	m_val = val;
}

void Neuron::activate()
{
	Scalar biasedVal = m_val + m_bias;

	m_actVal = 1.0 / (1.0 + exp(-biasedVal));
}

const Scalar& Neuron::getActVal() const
{
	return m_actVal;
}

void Neuron::calcDerivative()
{
	m_derivative = m_actVal * (1.0 - m_actVal);
}

void Neuron::calcLossDerivativeWithRespectToActFunc(const Scalar& desiredOutput)
{
	m_lossDerivativeWithRespectToActFunc = m_actVal - desiredOutput;
}

void Neuron::calcLossDerivativeWithRespectToActFunc(
	unsigned idxInLayer, 
	const std::vector<Neuron>& nextLayerNeurons, 
	const std::vector<std::vector<Synapse>>& outputSynapses)
{
	m_lossDerivativeWithRespectToActFunc = 0.0;

	for (int n = 0; n < nextLayerNeurons.size(); n++)
	{
		m_lossDerivativeWithRespectToActFunc +=
			outputSynapses[n][idxInLayer].getWeight() *
			nextLayerNeurons[n].getDerivative() *
			nextLayerNeurons[n].getLossDerivativeWithRespectToActFunc();
	}
}

const Scalar& Neuron::getDerivative() const
{
	return m_derivative;
}

const Scalar& Neuron::getLossDerivativeWithRespectToActFunc() const
{
	return m_lossDerivativeWithRespectToActFunc;
}

void Neuron::setBias(const Scalar& bias)
{
	m_bias = bias;
}

const Scalar& Neuron::getBias() const
{
	return m_bias;
}

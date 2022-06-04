#include "SynapsesMatrix.h"

SynapsesMatrix::SynapsesMatrix(
	unsigned nextLayerNeuronsCount, 
	unsigned previousLayerNeuronsCount)
{
	m_synapses.resize(nextLayerNeuronsCount);

	for (auto& it : m_synapses)
	{
		it.resize(previousLayerNeuronsCount);
	}
}

const std::pair<unsigned, unsigned>& SynapsesMatrix::getDimensions() const
{
	return { m_synapses.size(), m_synapses.back().size() };
}

const Scalar& SynapsesMatrix::getWeight(
	unsigned idxOfNextLayerNeuron, 
	unsigned idxOfPreviousLayerNeuron) const
{
	return m_synapses[idxOfNextLayerNeuron][idxOfPreviousLayerNeuron].getWeight();
}

void SynapsesMatrix::setWeight(
	unsigned idxOfNextLayerNeuron, 
	unsigned idxOfPreviousLayerNeuron,
	const Scalar& val)
{
	m_synapses[idxOfNextLayerNeuron][idxOfPreviousLayerNeuron].setWeight(val);
}

void SynapsesMatrix::updateWeightsGradients(
	const std::vector<Scalar>& input, 
	const std::vector<Neuron>& nextLayerNeurons)
{
	for (int n = 0; n < m_synapses.size(); n++)
	{
		for (int p = 0; p < m_synapses.back().size(); p++)
		{
			m_synapses[n][p].updateGradient(
				input[p],
				nextLayerNeurons[n].getDerivative(),
				nextLayerNeurons[n].getLossDerivativeWithRespectToActFunc()
			);
		}
	}
}

void SynapsesMatrix::updateWeightsGradients(
	const std::vector<Neuron>& previousLayerNeurons,
	const std::vector<Neuron>& nextLayerNeurons)
{
	for (int n = 0; n < m_synapses.size(); n++)
	{
		for (int p = 0; p < m_synapses.back().size(); p++)
		{
			m_synapses[n][p].updateGradient(
				previousLayerNeurons[p].getActVal(),
				nextLayerNeurons[n].getDerivative(),
				nextLayerNeurons[n].getLossDerivativeWithRespectToActFunc()
			);
		}
	}
}

const std::vector<std::vector<Synapse>>& SynapsesMatrix::getSynapsesMatrix() const
{
	return m_synapses;
}

void SynapsesMatrix::resetGradients()
{
	for (auto& it1 : m_synapses)
	{
		for (auto& it2 : it1)
		{
			it2.resetGradient();
		}
	}
}

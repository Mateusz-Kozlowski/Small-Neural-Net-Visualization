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
	unsigned idxOfPreviousLayerNeuron)
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

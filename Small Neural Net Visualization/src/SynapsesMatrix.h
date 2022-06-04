#pragma once

#include <vector>

#include "Neuron.h"

class SynapsesMatrix
{
public:
	SynapsesMatrix(
		unsigned nextLayerNeuronsCount, 
		unsigned previousLayerNeuronsCount
	);

	const std::pair<unsigned, unsigned>& getDimensions() const;

	const Scalar& getWeight(
		unsigned idxOfNextLayerNeuron,
		unsigned idxOfPreviousLayerNeuron
	) const;

	void setWeight(
		unsigned idxOfNextLayerNeuron, 
		unsigned idxOfPreviousLayerNeuron, 
		const Scalar& val
	);

	const std::vector<std::vector<Synapse>>& getSynapsesVector() const;

private:
	std::vector<std::vector<Synapse>> m_synapses;
};

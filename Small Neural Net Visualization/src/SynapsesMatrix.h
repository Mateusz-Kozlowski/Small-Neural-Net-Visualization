#pragma once

#include <vector>

#include "Synapse.h"

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
	);

	void setWeight(
		unsigned idxOfNextLayerNeuron, 
		unsigned idxOfPreviousLayerNeuron, 
		const Scalar& val
	);

private:
	std::vector<std::vector<Synapse>> m_synapses;
};

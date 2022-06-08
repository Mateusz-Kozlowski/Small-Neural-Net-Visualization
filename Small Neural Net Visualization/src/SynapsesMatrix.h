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

	void updateWeightsGradients(
		const std::vector<Scalar>& input,
		const std::vector<Neuron>& nextLayerNeurons
	);
	void updateWeightsGradients(
		const std::vector<Neuron>& previousLayerNeurons,
		const std::vector<Neuron>& nextLayerNeurons
	);

	const std::vector<std::vector<Synapse>>& getSynapsesMatrix() const;

	void resetGradients();

	void updateRendering(const Scalar& biggestAbsValOfWeightInNet);
	void render(sf::RenderTarget& target) const;

private:
	std::vector<std::vector<Synapse>> m_synapses;
};

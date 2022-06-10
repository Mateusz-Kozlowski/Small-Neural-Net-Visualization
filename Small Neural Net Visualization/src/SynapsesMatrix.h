#pragma once

#include "Neuron.h"

class SynapsesMatrix
{
public:
	SynapsesMatrix(
		const std::vector<Neuron>& nextLayerNeurons,
		unsigned inputSize,
		const std::vector<sf::CircleShape>& renderedNetInputsCircles,
		unsigned idxOfFirstRenderedNetInput,
		unsigned renderedInputsCount
	);
	SynapsesMatrix(
		const std::vector<Neuron>& nextLayerNeurons,
		const std::vector<Neuron>& previousLayerNeurons
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

	void updateRendering();
	void render(sf::RenderTarget& target) const;

private:
	Scalar getBiggestAbsValOfWeightInMatrix() const;

	std::vector<std::vector<Synapse>> m_synapses;
};

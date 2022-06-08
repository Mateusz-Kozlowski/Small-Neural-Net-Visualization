#include "SynapsesMatrix.h"

SynapsesMatrix::SynapsesMatrix(
	unsigned nextLayerNeuronsCount, 
	unsigned previousLayerNeuronsCount)
{
	m_synapses.resize(nextLayerNeuronsCount);

	for (auto& it : m_synapses)
	{
		it.resize(
			previousLayerNeuronsCount,
			Synapse(
				sf::Vector2f(0.0f, 0.0f),
				sf::Vector2f(0.0f, 0.0f)
			)
		);

		for (auto& it2 : it)
		{
			it2 = Synapse(
				sf::Vector2f(0.0f, 0.0f),
				sf::Vector2f(0.0f, 0.0f)
			);
		}
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
		// instead of n it could be for example 0 or .back() because it's a matrix (has rectangular shape):
		for (int p = 0; p < m_synapses[n].size(); p++)
		{
			m_synapses[n][p].updateGradient(
				p == 0 && n==0,
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
		// instead of n it could be for example 0 or .back() because it's a matrix (has rectangular shape)
		for (int p = 0; p < m_synapses[n].size(); p++)
		{
			m_synapses[n][p].updateGradient(
				false,
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

void SynapsesMatrix::updateRendering(const Scalar& biggestAbsValOfWeightInNet)
{
	for (auto& it : m_synapses)
	{
		for (auto& synapse : it)
		{
			synapse.updateRendering(biggestAbsValOfWeightInNet);
		}
	}
}

void SynapsesMatrix::render(sf::RenderTarget& target) const
{
	for (const auto& it : m_synapses)
	{
		for (const auto& synapse : it)
		{
			synapse.render(target);
		}
	}
}

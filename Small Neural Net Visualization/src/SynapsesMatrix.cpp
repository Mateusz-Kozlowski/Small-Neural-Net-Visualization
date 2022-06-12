#include "SynapsesMatrix.h"

SynapsesMatrix::SynapsesMatrix(
	const std::vector<Neuron>& nextLayerNeurons,
	unsigned inputSize,
	const std::vector<sf::CircleShape>& renderedNetInputsCircles,
	unsigned idxOfFirstRenderedNetInput,
	unsigned renderedInputsCount)
{
	m_synapses.resize(nextLayerNeurons.size());
	
	float neuronDiameter = nextLayerNeurons.back().getDiameter();

	for (int n = 0; n < m_synapses.size(); n++)
	{
		m_synapses[n].resize(
			inputSize,
			Synapse(
				false,
				sf::Vector2f(0.0f, 0.0f),
				sf::Vector2f(0.0f, 0.0f)
			)
		);

		for (int p = 0; p < m_synapses[n].size(); p++)
		{
			if (p >= idxOfFirstRenderedNetInput && p < idxOfFirstRenderedNetInput + renderedInputsCount)
			{
				const sf::CircleShape& renderedNetInputCircle = renderedNetInputsCircles[p - idxOfFirstRenderedNetInput];

				m_synapses[n][p] = Synapse(
					true,
					sf::Vector2f(
						renderedNetInputCircle.getPosition().x + neuronDiameter,
						renderedNetInputCircle.getPosition().y + neuronDiameter / 2.0f
					),
					sf::Vector2f(
						nextLayerNeurons[n].getPos().x,
						nextLayerNeurons[n].getPos().y + neuronDiameter / 2.0f
					)
				);
			}
			else
			{
				m_synapses[n][p] = Synapse(
					false,
					sf::Vector2f(0.0f, 0.0f),
					sf::Vector2f(0.0f, 0.0f)
				);
			}
		}
	}
}

SynapsesMatrix::SynapsesMatrix(
	const std::vector<Neuron>& nextLayerNeurons,
	const std::vector<Neuron>& previousLayerNeurons)
{
	m_synapses.resize(nextLayerNeurons.size());

	float neuronDiameter = nextLayerNeurons.back().getDiameter();

	for (int n = 0; n < m_synapses.size(); n++)
	{
		m_synapses[n].resize(
			previousLayerNeurons.size(),
			Synapse(
				true,
				sf::Vector2f(0.0f, 0.0f),
				sf::Vector2f(0.0f, 0.0f)
			)
		);

		for (int p = 0; p < m_synapses[n].size(); p++)
		{
			m_synapses[n][p] = Synapse(
				true,
				sf::Vector2f(
					previousLayerNeurons[p].getPos().x + neuronDiameter,
					previousLayerNeurons[p].getPos().y + neuronDiameter / 2.0f
				),
				sf::Vector2f(
					nextLayerNeurons[n].getPos().x,
					nextLayerNeurons[n].getPos().y + neuronDiameter / 2.0f
				)
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

void SynapsesMatrix::updateRendering()
{
	Scalar biggestAbsValOfWeightInMatrix = getBiggestAbsValOfWeightInMatrix();

	for (auto& it : m_synapses)
	{
		for (auto& synapse : it)
		{
			synapse.updateRendering(biggestAbsValOfWeightInMatrix);
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

Scalar SynapsesMatrix::getBiggestAbsValOfWeightInMatrix() const
{
	Scalar result = m_synapses.back().back().getWeight();

	for (const auto& it1 : m_synapses)
	{
		for (const auto& it2 : it1)
		{
			if (abs(it2.getWeight()) > result)
			{
				result = abs(it2.getWeight());
			}
		}
	}

	return result;
}

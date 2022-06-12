#include "Synapse.h"

Synapse::Synapse(bool isRendered, const sf::Vector2f& startPos, const sf::Vector2f& endPos)
	: m_isRendered(isRendered), m_gradient(0.0)
{
	m_weight = RandomEngine::getScalarInRange(-1.0, 1.0);

	m_line.resize(2);
	m_line[0].position = startPos;
	m_line[1].position = endPos;
}

const Scalar& Synapse::getWeight() const
{
	return m_weight;
}

void Synapse::setWeight(const Scalar& val)
{
	m_weight = val;
}

const Scalar& Synapse::getGradient() const
{
	return m_gradient;
}

void Synapse::resetGradient()
{
	m_gradient = 0.0;
}

void Synapse::updateGradient(
	const Scalar& previousActVal,
	const Scalar& nextDerivative,
	const Scalar& nextLossDerivativeWithRespectToActFunc)
{
	m_gradient += previousActVal * nextDerivative * nextLossDerivativeWithRespectToActFunc;
}

void Synapse::updateRendering(Scalar biggestAbsValOfWeightInMatrix)
{
	sf::Color color(0, 0, 0);

	if (m_weight > 0.0)
	{
		color.b = 255;
	}
	else
	{
		color.r = 255;
	}

	color.a = 255 * abs(m_weight) / biggestAbsValOfWeightInMatrix;

	m_line[0].color = color;
	m_line[1].color = color;
}

void Synapse::render(sf::RenderTarget& target) const
{
	if (!m_isRendered)
	{
		return;
	}

	target.draw(&m_line[0], m_line.size(), sf::Lines);
}

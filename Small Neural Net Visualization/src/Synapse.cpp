#include "Synapse.h"

Synapse::Synapse()
{
	m_weight = RandomEngine::getScalarInRange(-1.0, 1.0);
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

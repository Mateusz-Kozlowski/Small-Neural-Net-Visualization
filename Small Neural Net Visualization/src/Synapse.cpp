#include "Synapse.h"

Synapse::Synapse()
{
	m_weight = RandomEngine::getScalarInRange(-1.0, 1.0);
}

const Scalar& Synapse::getWeight()
{
	return m_weight;
}

void Synapse::setWeight(const Scalar& val)
{
	m_weight = val;
}

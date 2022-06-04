#pragma once

#include "RandomEngine.h"

class Synapse
{
public:
	Synapse();

	const Scalar& getWeight() const;

	void setWeight(const Scalar& val);

private:
	Scalar m_weight;
};

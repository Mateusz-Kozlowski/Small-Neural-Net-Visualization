#pragma once

#include "RandomEngine.h"

class Synapse
{
public:
	Synapse();

	const Scalar& getWeight();

	void setWeight(const Scalar& val);

private:
	Scalar m_weight;
};

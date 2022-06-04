#pragma once

#include "RandomEngine.h"

class Synapse
{
public:
	Synapse();

	const Scalar& getWeight() const;
	void setWeight(const Scalar& val);

	const Scalar& getGradient() const;
	void resetGradient();
	void updateGradient(
		const Scalar& previousActVal,
		const Scalar& nextDerivative,
		const Scalar& nextLossDerivativeWithRespectToActFunc
	);

private:
	Scalar m_weight;
	Scalar m_gradient;
};

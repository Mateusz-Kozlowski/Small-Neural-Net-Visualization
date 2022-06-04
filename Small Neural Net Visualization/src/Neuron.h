#pragma once

#include <vector>
#include <iostream>

#include "Synapse.h"
#include "Config.h"

class Neuron
{
public:
	void setVal(const Scalar& val);
	
	void activate();
	const Scalar& getActVal() const;
	
	void calcDerivative();

	void calcLossDerivativeWithRespectToActFunc(const Scalar& desiredOutput);
	void calcLossDerivativeWithRespectToActFunc(
		unsigned idxInLayer,
		const std::vector<Neuron>& nextLayerNeurons,
		const std::vector<std::vector<Synapse>>& outputSynapses
	);

	const Scalar& getDerivative() const;
	const Scalar& getLossDerivativeWithRespectToActFunc() const;

	void setBias(const Scalar& bias);
	const Scalar& getBias() const;
	
	const Scalar& getBiasGradient() const;
	void updateBiasGradient();
	void resetBiasGradient();

private:
	Scalar m_val;
	Scalar m_bias;
	Scalar m_actVal;
	Scalar m_derivative;
	Scalar m_lossDerivativeWithRespectToActFunc;
	Scalar m_biasGradient;
};

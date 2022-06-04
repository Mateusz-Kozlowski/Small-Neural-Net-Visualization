#pragma once

#include "Layer.h"

#include <iostream>

class OutputLayer : public Layer
{
public:
	OutputLayer(unsigned size);

	virtual void updateBiasesGradients() override;

	virtual unsigned getSize() const override;

	virtual void setBias(unsigned neuronIdx, const Scalar& bias) override;
	virtual void resetBiasesGradients() override;

	virtual void propagateForward(
		const Layer& previousLayer,
		const SynapsesMatrix& inputSynapses
	) override;

	virtual void calcDerivatives() override;
	virtual void calcErrors(const std::vector<Scalar>& desiredOutputs) override;
	virtual const Scalar& getActVal(unsigned neuronIdx) const override;
	virtual const Scalar& getBias(unsigned neuronIdx) const override;
	virtual const Scalar& getDerivative(unsigned neuronIdx) const override;
	virtual const Scalar& getLossDerivativeWithRespectToActFunc(unsigned neuronIdx) const override;

	const std::vector<Scalar>& getOutput();

	virtual const std::vector<Neuron>& getNeurons() const override;

private:
	std::vector<Neuron> m_neurons;
	std::vector<Scalar> m_output;
};

#pragma once

#include "Layer.h"

class HiddenLayer : public Layer
{
public:
	HiddenLayer(unsigned size);

	virtual unsigned getSize() const override;

	virtual void setBias(unsigned neuronIdx, const Scalar& bias) override;

	virtual void propagateForward(
		const std::vector<Scalar>& inputVector,
		const SynapsesMatrix& inputSynapses
	) override;
	virtual void propagateForward(
		const Layer& previousLayer,
		const SynapsesMatrix& inputSynapses
	) override;

	virtual void calcDerivatives() override;
	virtual const Scalar& getActVal(unsigned neuronIdx) const override;
	virtual const Scalar& getBias(unsigned neuronIdx) const override;
	virtual const Scalar& getDerivative(unsigned neuronIdx) const override;
	virtual const Scalar& getLossDerivativeWithRespectToActFunc(unsigned neuronIdx) const override;

	virtual const std::vector<Neuron>& getNeurons() const override;

	virtual void propagateErrorsBack(
		const Layer& nextLayer,
		const SynapsesMatrix& outputSynapses
	) override;

private:
	std::vector<Neuron> m_neurons;
};

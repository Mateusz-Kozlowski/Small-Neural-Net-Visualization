#pragma once

#include "SynapsesMatrix.h"

#include <functional>

class Layer
{
public:
	Layer(unsigned size);

	virtual void setInput(const std::vector<Scalar>& input) = 0;
	virtual const std::vector<Scalar>& getInput() const = 0;
	
	virtual void propagateForward(
		const std::vector<Scalar>& inputVector,
		const SynapsesMatrix& inputSynapses
	) = 0;
	virtual void propagateForward(
		const Layer& previousLayer,
		const SynapsesMatrix& inputSynapses
	) = 0;

	virtual const std::vector<Scalar>& getOutput() = 0;

	virtual void calcDerivatives() = 0;

	virtual void calcErrors(const std::vector<Scalar>& desiredOutputs) = 0;

	virtual void propagateErrorsBack(
		const Layer& nextLayer,
		const SynapsesMatrix& outputSynapses
	) = 0;

	virtual const std::vector<Neuron>& getNeurons() const = 0;
	
	virtual void updateBiasesGradients() = 0;

	virtual unsigned getSize() const = 0;

	virtual void setBias(unsigned neuronIdx, const Scalar& bias) = 0;
	virtual const Scalar& getBias(unsigned neuronIdx) const = 0;
	virtual void resetBiasesGradients() = 0;
};

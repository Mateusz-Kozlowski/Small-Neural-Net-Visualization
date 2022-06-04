#pragma once

#include "SynapsesMatrix.h"

class Layer
{
public:
	Layer(unsigned size);

	virtual void updateBiasesGradients();

	virtual const Scalar& getVal(unsigned idx) const;
	virtual unsigned getSize() const = 0;

	virtual const std::vector<Scalar>& getInput() const;

	virtual void setInput(const std::vector<Scalar>& input);
	virtual void setBias(unsigned neuronIdx, const Scalar& bias);
	virtual void resetBiasesGradients();

	virtual void propagateForward(
		const std::vector<Scalar>& inputVector,
		const SynapsesMatrix& inputSynapses
	);
	virtual void propagateForward(
		const Layer& previousLayer, 
		const SynapsesMatrix& inputSynapses
	);

	virtual void calcDerivatives();
	virtual void calcErrors(const std::vector<Scalar>& desiredOutputs);
	virtual void calcErrors(
		const std::vector<Neuron>& nextLayerNeurons,
		const std::vector<std::vector<Scalar>>& outputSynapsesWeights
	);
	virtual void propagateErrorsBack(
		const Layer& nextLayer, 
		const SynapsesMatrix& outputSynapses
	);
	virtual const Scalar& getActVal(unsigned neuronIdx) const;
	virtual const Scalar& getBias(unsigned neuronIdx) const;
	virtual const Scalar& getDerivative(unsigned neuronIdx) const;
	virtual const Scalar& getLossDerivativeWithRespectToActFunc(unsigned neuronIdx) const;

	virtual const std::vector<Neuron>& getNeurons() const;

	virtual const std::vector<Scalar>& getOutput();

protected:
	Scalar m_null;
	std::vector<Neuron> m_nullNeuronsVector;
	std::vector<Scalar> m_nullVector;
};

#pragma once

#include "Layer.h"

class NeuralLayer : public Layer
{
public:
	virtual void setInput(const std::vector<Scalar>& input) override;
	virtual const std::vector<Scalar>& getInput() const override;

	virtual void propagateForward(
		const std::vector<Scalar>& inputVector,
		const SynapsesMatrix& inputSynapses) = 0;

	virtual void propagateForward(
		const Layer& previousLayer,
		const SynapsesMatrix& inputSynapses
	) override;

	virtual const std::vector<Scalar>& getOutput() = 0;

	virtual void calcDerivatives() override;

	virtual void calcErrors(const std::vector<Scalar>& desiredOutputs) = 0;
	virtual void propagateErrorsBack(
		const Layer& nextLayer,
		const SynapsesMatrix& outputSynapses) = 0;

	virtual const std::vector<Neuron>& getNeurons() const override;

	virtual void updateBiasesGradients() override;

	virtual unsigned getSize() const override;

	virtual void setBias(unsigned neuronIdx, const Scalar& bias) override;
	virtual const Scalar& getBias(unsigned neuronIdx) const override;
	virtual void resetBiasesGradients() override;

	virtual void updateRendering() override;
	virtual void render(sf::RenderTarget& target, bool bgIsRendered) const override;

	virtual void moveVertically(float offset) override;

	virtual const std::vector<sf::CircleShape>& getRenderedInputsCircles() const override;

	virtual unsigned getIdxOfFirstRenderedNetInput() const override;
	virtual unsigned getNumberOfRenderedNetInputs() const override;

protected:
	void initNeurons(
		unsigned size,
		const sf::Vector2f& pos,
		float neuronCircleDiameter,
		float distBetweenNeuronsCircles,
		const sf::Color& baseNeuronsColor
	);
	void initBg(
		const sf::Vector2f& pos,
		const sf::Color& bgColor,
		float neuronCircleDiameter,
		float distBetweenNeuronsCircles
	);

	std::vector<Neuron> m_neurons;
};

#pragma once

#include "Layer.h"

class HiddenLayer : public Layer
{
public:
	HiddenLayer(
		unsigned size,
		const sf::Vector2f& pos,
		const sf::Color& bgColor,
		float renderedInputCircleDiameter,
		float distBetweenRenderedInputsCircles
	);

	virtual void setInput(const std::vector<Scalar>& input) override;
	virtual const std::vector<Scalar>& getInput() const override;

	virtual void propagateForward(
		const std::vector<Scalar>& inputVector,
		const SynapsesMatrix& inputSynapses
	) override;
	virtual void propagateForward(
		const Layer& previousLayer,
		const SynapsesMatrix& inputSynapses
	) override;

	virtual const std::vector<Scalar>& getOutput() override;

	virtual void calcDerivatives() override;

	virtual void calcErrors(const std::vector<Scalar>& desiredOutputs) override;

	virtual void propagateErrorsBack(
		const Layer& nextLayer,
		const SynapsesMatrix& outputSynapses
	) override;

	virtual const std::vector<Neuron>& getNeurons() const override;

	virtual void updateBiasesGradients() override;

	virtual unsigned getSize() const override;

	virtual void setBias(unsigned neuronIdx, const Scalar& bias) override;
	virtual const Scalar& getBias(unsigned neuronIdx) const override;
	virtual void resetBiasesGradients() override;

	virtual void updateRendering() override;
	virtual void render(sf::RenderTarget& target) const override;

private:
	void initNeurons(
		unsigned size,
		const sf::Vector2f& pos,
		float renderedInputCircleDiameter,
		float distBetweenRenderedInputsCircles
	);
	void initBg(
		const sf::Vector2f& pos,
		const sf::Color& bgColor,
		float renderedInputCircleDiameter,
		float distBetweenRenderedInputsCircles
	);

	std::vector<Neuron> m_neurons;

	sf::RectangleShape m_bg;
};

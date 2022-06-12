#pragma once

#include "Synapse.h"

class Neuron
{
public:
	Neuron(
		const sf::Vector2f& pos,
		float radius,
		const sf::Color& baseColor
	);

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

	void updateRendering();
	void render(sf::RenderTarget& target) const;

	const sf::Vector2f& getPos() const;
	void setPos(const sf::Vector2f& pos);

	float getDiameter() const;

	const sf::Color& getBaseColor() const;

private:
	void initCircle(const sf::Vector2f& pos, float radius);

	Scalar m_val;
	Scalar m_bias;
	Scalar m_actVal;
	Scalar m_derivative;
	Scalar m_lossDerivativeWithRespectToActFunc;
	Scalar m_biasGradient;

	sf::Color m_baseColor;

	sf::CircleShape m_circle;
};

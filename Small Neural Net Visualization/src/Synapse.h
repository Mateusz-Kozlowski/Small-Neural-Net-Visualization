#pragma once

#include "RandomEngine.h"

#include "SFML/Graphics.hpp"

class Synapse
{
public:
	Synapse(bool isRendered, const sf::Vector2f& startPos, const sf::Vector2f& endPos);

	const Scalar& getWeight() const;
	void setWeight(const Scalar& val);

	const Scalar& getGradient() const;
	void resetGradient();
	void updateGradient(
		bool flag,
		const Scalar& previousActVal,
		const Scalar& nextDerivative,
		const Scalar& nextLossDerivativeWithRespectToActFunc
	);

	void updateRendering(const Scalar& biggestAbsValOfWeightInNet);
	void render(sf::RenderTarget& target) const;

private:
	bool m_isRendered;

	Scalar m_weight;
	Scalar m_gradient;

	std::vector<sf::Vertex> m_line;
};

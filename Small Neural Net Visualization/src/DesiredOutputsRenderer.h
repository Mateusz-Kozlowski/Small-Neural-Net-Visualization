#pragma once

#include "Config.h"

class DesiredOutputsRenderer
{
public:
	DesiredOutputsRenderer(
		unsigned size,
		const sf::Vector2f& pos,
		const sf::Color& bgColor,
		float diameterOfDesiredOutputCircle,
		float distBetweenDesiredOutputCircles,
		const sf::Color& desiredOutputsCirclesColor
	);

	void setDesiredOutput(const std::vector<Scalar>& desiredOutput);

	void render(sf::RenderTarget& target, bool bgIsRendered) const;

private:
	void initCircles(
		unsigned size,
		const sf::Vector2f& pos,
		float diameterOfDesiredOutputCircle,
		float distBetweenDesiredOutputCircles
	);
	void initBg(
		const sf::Vector2f& pos,
		const sf::Color& bgColor,
		float diameterOfDesiredOutputCircle,
		float distBetweenDesiredOutputCircles
	);

	std::vector<sf::CircleShape> m_desiredOutputsCircles;

	sf::RectangleShape m_bg;

	sf::Color m_desiredOutputsCirclesColor;
};

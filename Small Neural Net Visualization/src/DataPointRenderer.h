#pragma once

#include "SFML/Graphics.hpp"
#include "SFML/Audio.hpp"
#include "SFML/Network.hpp"
#include "SFML/Window.hpp"
#include "SFML/System.hpp"

#include <vector>

#include "config.h"

class DataPointRenderer
{
public:
	DataPointRenderer(
		sf::Vector2f pos,
		sf::Vector2f size, 
		sf::Color outlineColor,
		float outlineThickness,
		unsigned pixelsCount
	);

	void updateRendering(const std::vector<Scalar>& dataPoint);
	void render(sf::RenderTarget& target) const;

private:
	void initOutline(
		sf::Vector2f pos,
		sf::Vector2f size,
		sf::Color outlineColor,
		float outlineThickness
	);
	void initDataPointRectangles();

	unsigned m_pixelsCount;

	sf::RectangleShape m_outline;
	std::vector<sf::RectangleShape> m_dataPointRectangles;
};

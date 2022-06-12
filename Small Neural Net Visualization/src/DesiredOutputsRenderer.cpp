#include "DesiredOutputsRenderer.h"

DesiredOutputsRenderer::DesiredOutputsRenderer(
	unsigned size,
	const sf::Vector2f& pos,
	const sf::Color& bgColor,
	float diameterOfDesiredOutputCircle,
	float distBetweenDesiredOutputCircles,
	const sf::Color& desiredOutputsCirclesColor)
	: m_desiredOutputsCirclesColor(desiredOutputsCirclesColor)
{
	initCircles(
		size, 
		pos, 
		diameterOfDesiredOutputCircle,
		distBetweenDesiredOutputCircles
	);
	initBg(
		pos, 
		bgColor, 
		diameterOfDesiredOutputCircle, 
		distBetweenDesiredOutputCircles
	);
}

const sf::Color& DesiredOutputsRenderer::getBgColor() const
{
	return m_bg.getFillColor();
}

void DesiredOutputsRenderer::setDesiredOutput(const std::vector<Scalar>& desiredOutput)
{
	for (int i = 0; i < desiredOutput.size(); i++)
	{
		sf::Color desiredOutputsCirclesColor = m_desiredOutputsCirclesColor;
		
		desiredOutputsCirclesColor.a = 255 * desiredOutput[i];

		m_desiredOutputsCircles[i].setFillColor(desiredOutputsCirclesColor);
	}
}

void DesiredOutputsRenderer::render(sf::RenderTarget& target, bool bgIsRendered) const
{
	if (bgIsRendered)
	{
		target.draw(m_bg);
	}
	
	for (const auto& desiredOutputCircle : m_desiredOutputsCircles)
	{
		target.draw(desiredOutputCircle);
	}
}

void DesiredOutputsRenderer::initCircles(
	unsigned size, 
	const sf::Vector2f& pos, 
	float diameterOfDesiredOutputCircle, 
	float distBetweenDesiredOutputCircles)
{
	for (int i = 0; i < size; i++)
	{
		m_desiredOutputsCircles.emplace_back(
			sf::CircleShape(
				diameterOfDesiredOutputCircle / 2.0f
			)
		);

		m_desiredOutputsCircles.back().setPosition(
			sf::Vector2f(
				pos.x,
				pos.y + i * (diameterOfDesiredOutputCircle + distBetweenDesiredOutputCircles)
			)
		);
	}
}

void DesiredOutputsRenderer::initBg(
	const sf::Vector2f& pos, 
	const sf::Color& bgColor,
	float diameterOfDesiredOutputCircle,
	float distBetweenDesiredOutputCircles)
{
	m_bg.setPosition(pos);
	m_bg.setFillColor(bgColor);
	m_bg.setSize(
		sf::Vector2f(
			diameterOfDesiredOutputCircle,
			m_desiredOutputsCircles.size() * (diameterOfDesiredOutputCircle + distBetweenDesiredOutputCircles) - distBetweenDesiredOutputCircles
		)
	);
}

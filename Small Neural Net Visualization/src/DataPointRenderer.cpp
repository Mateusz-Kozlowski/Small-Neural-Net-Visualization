#include "DataPointRenderer.h"

DataPointRenderer::DataPointRenderer(
	sf::Vector2f pos, 
	sf::Vector2f size, 
	sf::Color outlineColor,
	float outlineThickness,
	unsigned pixelsCount)
	: m_pixelsCount(pixelsCount)
{
	initOutline(pos, size, outlineColor, outlineThickness);
	initDataPointRectangles();
}

void DataPointRenderer::updateRendering(const std::vector<Scalar>& dataPoint)
{
	for (int i = 0; i < m_pixelsCount; i++)
	{
		unsigned color = dataPoint[i] * 255.0;
		
		m_dataPointRectangles[i].setFillColor(sf::Color(color, color, color));
	}
}

void DataPointRenderer::render(sf::RenderTarget& target) const
{
	target.draw(m_outline);

	for (const auto& it : m_dataPointRectangles)
	{
		target.draw(it);
	}
}

void DataPointRenderer::initOutline(
	sf::Vector2f pos, 
	sf::Vector2f size, 
	sf::Color outlineColor,
	float outlineThickness)
{
	m_outline.setPosition(
		sf::Vector2f(
			pos.x + outlineThickness,
			pos.y + outlineThickness
		)
	);
	m_outline.setSize(size);
	m_outline.setOutlineThickness(outlineThickness);
	m_outline.setOutlineColor(outlineColor);
}

void DataPointRenderer::initDataPointRectangles()
{
	m_dataPointRectangles.resize(m_pixelsCount);

	float pixelWidth = m_outline.getSize().x / sqrt(m_pixelsCount);
	float pixelHeight = m_outline.getSize().y / sqrt(m_pixelsCount);

	for (int i = 0; i < sqrt(m_pixelsCount); i++)
	{
		for (int j = 0; j < sqrt(m_pixelsCount); j++)
		{
			m_dataPointRectangles[i * sqrt(m_pixelsCount) + j].setPosition(
				sf::Vector2f(
					m_outline.getPosition().x + pixelWidth * j,
					m_outline.getPosition().y + pixelHeight * i
				)
			);
			m_dataPointRectangles[i * sqrt(m_pixelsCount) + j].setSize(
				sf::Vector2f(
					pixelWidth,
					pixelHeight
				)
			);
		}
	}
}

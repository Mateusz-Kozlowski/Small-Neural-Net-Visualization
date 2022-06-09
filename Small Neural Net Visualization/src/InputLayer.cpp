#include "InputLayer.h"

InputLayer::InputLayer(
	unsigned size,
	const sf::Vector2f& pos,
	const sf::Color& bgColor,
	unsigned firstRenderedInputIdx,
	unsigned renderedInputsCount,
	float renderedInputCircleDiameter,
	float distBetweenRenderedInputsCircles)
	: m_firstRenderedInputIdx(firstRenderedInputIdx)
{
	m_input.resize(size);
	initRenderedInputsCircles(
		pos,
		renderedInputsCount,
		renderedInputCircleDiameter,
		distBetweenRenderedInputsCircles
	);
	initBg(
		pos,
		bgColor,
		renderedInputsCount,
		renderedInputCircleDiameter,
		distBetweenRenderedInputsCircles
	);
}

void InputLayer::setInput(const std::vector<Scalar>& input)
{
	if (input.size() != m_input.size())
	{
		std::cerr << "InputLayer::setInput(const std::vector<Scalar>&): " << "WRONG INPUT SIZE!\n";
		exit(-13);
	}
	
	for (int i = 0; i < input.size(); i++)
	{
		m_input[i] = input[i];
	}
}

const std::vector<Scalar>& InputLayer::getInput() const
{
	return m_input;
}

void InputLayer::propagateForward(
	const std::vector<Scalar>& inputVector, 
	const SynapsesMatrix& inputSynapses)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::propagateForward(
	const Layer& previousLayer, 
	const SynapsesMatrix& inputSynapses)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Scalar>& InputLayer::getOutput()
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::calcDerivatives()
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::propagateErrorsBack(
	const Layer& nextLayer, 
	const SynapsesMatrix& outputSynapses)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Neuron>& InputLayer::getNeurons() const
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::updateBiasesGradients()
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

unsigned InputLayer::getSize() const
{
	return m_input.size();
}

void InputLayer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const Scalar& InputLayer::getBias(unsigned neuronIdx) const
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::resetBiasesGradients()
{
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::updateRendering()
{
	for (int i = 0; i < m_renderedInputsCircles.size(); i++)
	{
		sf::Color color(255, 255, 255);

		color.a = 255 * m_input[m_firstRenderedInputIdx + i];

		m_renderedInputsCircles[i].setFillColor(color);
	}
}

void InputLayer::render(sf::RenderTarget& target) const
{
	target.draw(m_bg);
	
	for (const auto& renderedInputCircle : m_renderedInputsCircles)
	{
		target.draw(renderedInputCircle);
	}
}

void InputLayer::initRenderedInputsCircles(
	const sf::Vector2f& pos,
	unsigned renderedInputsCount,
	float renderedInputCircleDiameter,
	float distBetweenRenderedInputsCircles)
{
	m_renderedInputsCircles.resize(
		renderedInputsCount, 
		sf::CircleShape(renderedInputCircleDiameter / 2.0f)
	);

	for (int i = 0; i < renderedInputsCount; i++)
	{
		m_renderedInputsCircles[i].setPosition(
			sf::Vector2f(
				pos.x,
				pos.y + i * (renderedInputCircleDiameter + distBetweenRenderedInputsCircles)
			)
		);
	}
}

void InputLayer::initBg(
	const sf::Vector2f& pos,
	const sf::Color& bgColor,
	unsigned renderedInputsCount, 
	float renderedInputCircleDiameter, 
	float distBetweenRenderedInputsCircles)
{
	m_bg.setPosition(pos);
	m_bg.setFillColor(bgColor);
	m_bg.setSize(
		sf::Vector2f(
			renderedInputCircleDiameter,
			m_renderedInputsCircles.size() * (renderedInputCircleDiameter + distBetweenRenderedInputsCircles) - distBetweenRenderedInputsCircles
		)
	);
}

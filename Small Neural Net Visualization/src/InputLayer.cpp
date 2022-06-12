#include "InputLayer.h"

InputLayer::InputLayer(
	unsigned size,
	const sf::Vector2f& pos,
	const sf::Color& bgColor,
	unsigned firstRenderedInputIdx,
	unsigned renderedInputsCount,
	float renderedInputCircleDiameter,
	float distBetweenRenderedInputsCircles,
	const sf::Color& renderedInputCirclesBaseColor)
	: m_firstRenderedInputIdx(firstRenderedInputIdx),
	  m_renderedInputCirclesBaseColor(renderedInputCirclesBaseColor)
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
		std::cerr << input.size() << "!=" << m_input.size() << '\n';
		exit(-13);
	}
	
	m_input = input;
}

const std::vector<Scalar>& InputLayer::getInput() const
{
	return m_input;
}

void InputLayer::propagateForward(
	const std::vector<Scalar>& inputVector, 
	const SynapsesMatrix& inputSynapses)
{
	std::cerr << "InputLayer::propagateForward(const std::vector<Scalar>&, const SynapsesMatrix&):\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::propagateForward(
	const Layer& previousLayer, 
	const SynapsesMatrix& inputSynapses)
{
	std::cerr << "InputLayer::propagateForward(const Layer&, const SynapsesMatrix&):\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Scalar>& InputLayer::getOutput()
{
	std::cerr << "const std::vector<Scalar>& InputLayer::getOutput():\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::calcDerivatives()
{
	std::cerr << "void InputLayer::calcDerivatives():\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	std::cerr << "void InputLayer::calcErrors(const std::vector<Scalar>&):\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::propagateErrorsBack(
	const Layer& nextLayer, 
	const SynapsesMatrix& outputSynapses)
{
	std::cerr << "void InputLayer::propagateErrorsBack(const Layer&, const SynapsesMatrix&):\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Neuron>& InputLayer::getNeurons() const
{
	std::cerr << "const std::vector<Neuron>& InputLayer::getNeurons() const:\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::updateBiasesGradients()
{
	std::cerr << "void InputLayer::updateBiasesGradients():\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

unsigned InputLayer::getSize() const
{
	return m_input.size();
}

void InputLayer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	std::cerr << "void InputLayer::setBias(unsigned, const Scalar&):\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const Scalar& InputLayer::getBias(unsigned neuronIdx) const
{
	std::cerr << "const Scalar& InputLayer::getBias(unsigned) const:\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::resetBiasesGradients()
{
	std::cerr << "void InputLayer::resetBiasesGradients():\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void InputLayer::updateRendering()
{
	for (int i = 0; i < m_renderedInputsCircles.size(); i++)
	{
		sf::Color renderedInputCirclesBaseColor = m_renderedInputCirclesBaseColor;

		renderedInputCirclesBaseColor.a = 255 * m_input[m_firstRenderedInputIdx + i];

		m_renderedInputsCircles[i].setFillColor(renderedInputCirclesBaseColor);
	}
}

void InputLayer::render(sf::RenderTarget& target, bool bgIsRendered) const
{
	if (bgIsRendered)
	{
		target.draw(m_bg);
	}
	
	for (const auto& renderedInputCircle : m_renderedInputsCircles)
	{
		target.draw(renderedInputCircle);
	}
}

void InputLayer::moveVertically(float offset)
{
	std::cerr << "void InputLayer::moveVertically(float):\n";
	std::cerr << "InputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<sf::CircleShape>& InputLayer::getRenderedInputsCircles() const
{
	return m_renderedInputsCircles;
}

unsigned InputLayer::getIdxOfFirstRenderedNetInput() const
{
	return m_firstRenderedInputIdx;
}

unsigned InputLayer::getNumberOfRenderedNetInputs() const
{
	return m_renderedInputsCircles.size();
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

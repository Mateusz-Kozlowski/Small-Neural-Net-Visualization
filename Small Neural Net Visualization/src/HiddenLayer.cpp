#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(
	unsigned size,
	const sf::Vector2f& pos,
	const sf::Color& bgColor,
	float neuronCircleDiameter,
	float distBetweenNeuronsCircles)
{
	initNeurons(
		size, 
		pos, 
		neuronCircleDiameter,
		distBetweenNeuronsCircles
	);
	initBg(
		pos,
		bgColor,
		neuronCircleDiameter,
		distBetweenNeuronsCircles
	);
}

void HiddenLayer::setInput(const std::vector<Scalar>& input)
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Scalar>& HiddenLayer::getInput() const
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void HiddenLayer::propagateForward(
	const std::vector<Scalar>& inputVector,
	const SynapsesMatrix& inputSynapses)
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		Scalar input = 0.0;
		for (int p = 0; p < inputVector.size(); p++)
		{
			input += inputVector[p] * inputSynapses.getWeight(i, p);
		}

		m_neurons[i].setVal(input);
		m_neurons[i].activate();
	}
}

void HiddenLayer::propagateForward(
	const Layer& previousLayer,
	const SynapsesMatrix& inputSynapses)
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		Scalar input = 0.0;
		for (int p = 0; p < previousLayer.getNeurons().size(); p++)
		{
			input += previousLayer.getNeurons()[p].getActVal() * inputSynapses.getWeight(i, p);
		}

		m_neurons[i].setVal(input);
		m_neurons[i].activate();
	}
}

const std::vector<Scalar>& HiddenLayer::getOutput()
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void HiddenLayer::calcDerivatives()
{
	for (auto& neuron : m_neurons)
	{
		neuron.calcDerivative();
	}
}

void HiddenLayer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void HiddenLayer::propagateErrorsBack(
	const Layer& nextLayer,
	const SynapsesMatrix& outputSynapses)
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].calcLossDerivativeWithRespectToActFunc(
			i,
			nextLayer.getNeurons(),
			outputSynapses.getSynapsesMatrix()
		);
	}
}

const std::vector<Neuron>& HiddenLayer::getNeurons() const
{
	return m_neurons;
}

void HiddenLayer::updateBiasesGradients()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].updateBiasGradient();
	}
}

unsigned HiddenLayer::getSize() const
{
	return m_neurons.size();
}

void HiddenLayer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	m_neurons[neuronIdx].setBias(bias);
}

const Scalar& HiddenLayer::getBias(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getBias();
}

void HiddenLayer::resetBiasesGradients()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].resetBiasGradient();
	}
}

void HiddenLayer::updateRendering()
{
	for (auto& neuron : m_neurons)
	{
		neuron.updateRendering();
	}
}

void HiddenLayer::render(sf::RenderTarget& target, bool bgIsRendered) const
{
	if (bgIsRendered)
	{
		target.draw(m_bg);
	}

	for (auto& neuron : m_neurons)
	{
		neuron.render(target);
	}
}

void HiddenLayer::moveVertically(float offset)
{
	m_bg.setPosition(
		m_bg.getPosition().x,
		m_bg.getPosition().y + offset
	);

	for (auto& neuron : m_neurons)
	{
		neuron.setPos(
			sf::Vector2f(
				neuron.getPos().x,
				neuron.getPos().y + offset
			)
		);
	}
}

const std::vector<sf::CircleShape>& HiddenLayer::getRenderedInputsCircles() const
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

unsigned HiddenLayer::getIdxOfFirstRenderedNetInput() const
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

unsigned HiddenLayer::getNumberOfRenderedNetInputs() const
{
	std::cerr << "HiddenLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void HiddenLayer::initNeurons(
	unsigned size,
	const sf::Vector2f& pos,
	float neuronCircleDiameter,
	float distBetweenNeuronsCircles)
{
	for (int i = 0; i < size; i++)
	{
		m_neurons.emplace_back(
			Neuron(
				sf::Vector2f(
					pos.x,
					pos.y + i * (neuronCircleDiameter + distBetweenNeuronsCircles)
				),
				neuronCircleDiameter / 2.0f
			)
		);
	}
}

void HiddenLayer::initBg(
	const sf::Vector2f& pos,
	const sf::Color& bgColor,
	float neuronCircleDiameter,
	float distBetweenNeuronsCircles)
{
	m_bg.setPosition(pos);
	m_bg.setFillColor(bgColor);
	m_bg.setSize(
		sf::Vector2f(
			neuronCircleDiameter,
			m_neurons.size() * (neuronCircleDiameter + distBetweenNeuronsCircles) - distBetweenNeuronsCircles
		)
	);
}

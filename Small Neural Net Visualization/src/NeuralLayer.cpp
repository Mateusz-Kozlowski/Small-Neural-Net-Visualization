#include "NeuralLayer.h"

void NeuralLayer::setInput(const std::vector<Scalar>& input)
{
	std::cerr << "NeuralLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Scalar>& NeuralLayer::getInput() const
{
	std::cerr << "NeuralLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void NeuralLayer::propagateForward(
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

void NeuralLayer::calcDerivatives()
{
	for (auto& neuron : m_neurons)
	{
		neuron.calcDerivative();
	}
}

const std::vector<Neuron>& NeuralLayer::getNeurons() const
{
	return m_neurons;
}

void NeuralLayer::updateBiasesGradients()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].updateBiasGradient();
	}
}

unsigned NeuralLayer::getSize() const
{
	return m_neurons.size();
}

void NeuralLayer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	m_neurons[neuronIdx].setBias(bias);
}

const Scalar& NeuralLayer::getBias(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getBias();
}

void NeuralLayer::resetBiasesGradients()
{
	for (auto& neuron : m_neurons)
	{
		neuron.resetBiasGradient();
	}
}

void NeuralLayer::updateRendering()
{
	for (auto& neuron : m_neurons)
	{
		neuron.updateRendering();
	}
}

void NeuralLayer::render(sf::RenderTarget& target, bool bgIsRendered) const
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

void NeuralLayer::moveVertically(float offset)
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

const std::vector<sf::CircleShape>& NeuralLayer::getRenderedInputsCircles() const
{
	std::cerr << "NeuralLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

unsigned NeuralLayer::getIdxOfFirstRenderedNetInput() const
{
	std::cerr << "NeuralLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

unsigned NeuralLayer::getNumberOfRenderedNetInputs() const
{
	std::cerr << "NeuralLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void NeuralLayer::initNeurons(
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

void NeuralLayer::initBg(
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

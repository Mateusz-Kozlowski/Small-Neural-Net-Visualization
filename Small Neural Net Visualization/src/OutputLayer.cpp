#include "OutputLayer.h"

OutputLayer::OutputLayer(
	unsigned size,
	const sf::Vector2f& pos,
	const sf::Color& bgColor,
	float renderedInputCircleDiameter,
	float distBetweenRenderedInputsCircles)
{
	initNeurons(
		size,
		pos, 
		renderedInputCircleDiameter, 
		distBetweenRenderedInputsCircles
	);
	initBg(
		pos, 
		bgColor,
		renderedInputCircleDiameter, 
		distBetweenRenderedInputsCircles
	);
	m_output.resize(size);	
}

void OutputLayer::setInput(const std::vector<Scalar>& input)
{
	std::cerr << "OutputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Scalar>& OutputLayer::getInput() const
{
	std::cerr << "OutputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void OutputLayer::propagateForward(
	const std::vector<Scalar>& inputVector,
	const SynapsesMatrix& inputSynapses)
{
	std::cerr << "OutputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

void OutputLayer::propagateForward(
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

const std::vector<Scalar>& OutputLayer::getOutput()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_output[i] = m_neurons[i].getActVal();
	}

	return m_output;
}

void OutputLayer::calcDerivatives()
{
	for (auto& neuron : m_neurons)
	{
		neuron.calcDerivative();
	}
}

void OutputLayer::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	for (int i = 0; i < desiredOutputs.size(); i++)
	{
		m_neurons[i].calcLossDerivativeWithRespectToActFunc(desiredOutputs[i]);
	}
}

void OutputLayer::propagateErrorsBack(
	const Layer& nextLayer,
	const SynapsesMatrix& outputSynapses)
{
	std::cerr << "OutputLayer class doesn't support this function\n";
	throw std::bad_function_call();
}

const std::vector<Neuron>& OutputLayer::getNeurons() const
{
	return m_neurons;
}

void OutputLayer::updateBiasesGradients()
{
	for (int i = 0; i < m_neurons.size(); i++)
	{
		m_neurons[i].updateBiasGradient();
	}
}

unsigned OutputLayer::getSize() const
{
	return m_neurons.size();
}

void OutputLayer::setBias(unsigned neuronIdx, const Scalar& bias)
{
	m_neurons[neuronIdx].setBias(bias);
}

const Scalar& OutputLayer::getBias(unsigned neuronIdx) const
{
	return m_neurons[neuronIdx].getBias();
}

void OutputLayer::resetBiasesGradients()
{
	for (auto& neuron : m_neurons)
	{
		neuron.resetBiasGradient();
	}
}

void OutputLayer::updateRendering()
{
	for (auto& neuron : m_neurons)
	{
		neuron.updateRendering();
	}
}

void OutputLayer::render(sf::RenderTarget& target) const
{
	target.draw(m_bg);

	for (auto& neuron : m_neurons)
	{
		neuron.render(target);
	}
}

void OutputLayer::moveVertically(float offset)
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

void OutputLayer::initNeurons(
	unsigned size,
	const sf::Vector2f& pos, 
	float renderedInputCircleDiameter, 
	float distBetweenRenderedInputsCircles)
{
	for (int i = 0; i < size; i++)
	{
		m_neurons.emplace_back(
			Neuron(
				sf::Vector2f(
					pos.x,
					pos.y + i * (renderedInputCircleDiameter + distBetweenRenderedInputsCircles)
				),
				renderedInputCircleDiameter / 2.0f
			)
		);
	}
}

void OutputLayer::initBg(
	const sf::Vector2f& pos, 
	const sf::Color& bgColor,
	float renderedInputCircleDiameter, 
	float distBetweenRenderedInputsCircles)
{
	m_bg.setPosition(pos);
	m_bg.setFillColor(bgColor);
	m_bg.setSize(
		sf::Vector2f(
			renderedInputCircleDiameter,
			m_neurons.size() * (renderedInputCircleDiameter + distBetweenRenderedInputsCircles) - distBetweenRenderedInputsCircles
		)
	);
}

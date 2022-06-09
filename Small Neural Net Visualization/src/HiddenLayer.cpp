#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(
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
			if (inputVector[p] * inputSynapses.getWeight(i, p) != 0.0)
			{
				/*std::cout
					<< p << ": "
					<< "val += "
					<< inputVector[p] << "*" << inputSynapses.getWeight(i, p)
					<< inputVector[p] * inputSynapses.getWeight(i, p) << '\n';*/
			}

			//Scalar itsStupidButItPreventAbug = inputVector[p] * inputSynapses.getWeight(i, p);

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

void HiddenLayer::render(sf::RenderTarget& target) const
{
	target.draw(m_bg);

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

void HiddenLayer::initNeurons(
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

void HiddenLayer::initBg(
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

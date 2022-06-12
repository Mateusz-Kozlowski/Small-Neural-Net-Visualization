#include "NeuralNet.h"

NeuralNet::NeuralNet(
	const std::vector<unsigned>& topology,
	const Scalar& learningRate,
	unsigned miniBatchSize,
	const sf::Vector2f& pos,
	const sf::Vector2f& size,
	const sf::Color& bgColor,
	const sf::Color& layersBgColor,
	const sf::Color& desiredOutputsBgColor,
	const sf::Color& baseNeuronsColor,
	bool bgIsRendered,
	bool layersBgAreRendered)
	: m_trainingStep(0U),
	  m_learningRate(learningRate), 
	  m_miniBatchSize(miniBatchSize),
	  m_bgIsRendered(bgIsRendered),
	  m_layersBgAreRendered(layersBgAreRendered)
{
	initLayers(topology, pos, size, layersBgColor, baseNeuronsColor);
	initSynapses(topology);
	initBg(pos, size, bgColor);
	initDesiredOutputsRenderer(
		topology, 
		pos, 
		size, 
		desiredOutputsBgColor,
		baseNeuronsColor
	);
}

const sf::Vector2f& NeuralNet::getPos() const
{
	return m_bg.getPosition();
}

const sf::Vector2f& NeuralNet::getSize() const
{
	return m_bg.getSize();
}

void NeuralNet::save(const std::string& path)
{
	std::ofstream file(path);

	if (!file.is_open())
	{
		std::cerr << "CANNOT OPEN: " << path << '\n';
		exit(-100);
	}

	file << m_trainingStep << '\n';
	file << m_learningRate << '\n';
	file << m_miniBatchSize << '\n';
	file << m_bgIsRendered << '\n';
	file << m_layersBgAreRendered << '\n';
	file << m_bg.getPosition().x << ' ' << m_bg.getPosition().y << '\n';
	file << m_bg.getSize().x << ' ' << m_bg.getSize().y << '\n';

	file << static_cast<int>(m_bg.getFillColor().r) << ' ';
	file << static_cast<int>(m_bg.getFillColor().g) << ' ';
	file << static_cast<int>(m_bg.getFillColor().b) << ' ';
	file << static_cast<int>(m_bg.getFillColor().a) << '\n';
	
	file << static_cast<int>(m_layers.back()->getBgColor().r) << ' ';
	file << static_cast<int>(m_layers.back()->getBgColor().g) << ' ';
	file << static_cast<int>(m_layers.back()->getBgColor().b) << ' ';
	file << static_cast<int>(m_layers.back()->getBgColor().a) << '\n';

	file << static_cast<int>(m_desiredOutputsRenderer->getBgColor().r) << ' ';
	file << static_cast<int>(m_desiredOutputsRenderer->getBgColor().g) << ' ';
	file << static_cast<int>(m_desiredOutputsRenderer->getBgColor().b) << ' ';
	file << static_cast<int>(m_desiredOutputsRenderer->getBgColor().a) << '\n';

	file << static_cast<int>(m_layers.back()->getNeurons().back().getBaseColor().r) << ' ';
	file << static_cast<int>(m_layers.back()->getNeurons().back().getBaseColor().g) << ' ';
	file << static_cast<int>(m_layers.back()->getNeurons().back().getBaseColor().b) << ' ';
	file << static_cast<int>(m_layers.back()->getNeurons().back().getBaseColor().a) << '\n';

	// save topology:
	file << m_layers.size() << '\n';
	for (int i = 0; i < m_layers.size(); i++)
	{
		file << m_layers[i]->getSize() << ' ';
	}
	file << '\n';

	// save biases:
	for (int i=1; i<m_layers.size(); i++)
	{
		for (int j=0; j<m_layers[i]->getSize(); j++)
		{
			file << m_layers[i]->getNeurons()[j].getBias() << ' ';
		}
	}
	file << '\n';

	// save weights:
	for (const auto& synapsesMatrix : m_synapses)
	{
		for (const auto& it1 : synapsesMatrix.getSynapsesMatrix())
		{
			for (const auto& it2 : it1)
			{
				file << it2.getWeight() << '\n';
			}
		}
	}
	
	file.close();
}

void NeuralNet::load(const std::string& path)
{
	std::ifstream file(path);

	if (!file.is_open())
	{
		std::cerr << "CANNOT OPEN: " << path << '\n';
		exit(-13);
	}

	sf::Vector2f pos;
	sf::Vector2f size;
	
	unsigned bgColorRed, bgColorGreen, bgColorBlue, bgColorAlfa;
	unsigned layersBgColorRed, layersBgColorGreen, layersBgColorBlue, layersBgColorAlfa;
	unsigned desiredOutputBgColorRed, desiredOutputBgColorGreen, desiredOutputBgColorBlue, desiredOutputBgColorAlfa;
	unsigned baseNeuronsColorRed, baseNeuronsColorGreen, baseNeuronsColorBlue, baseNeuronsColorAlfa;
	sf::Color bgColor;
	sf::Color layersBgColor;
	sf::Color desiredOutputBgColor;
	sf::Color baseNeuronsColor;

	unsigned layersCount;
	std::vector<unsigned> topology;

	file >> m_trainingStep;
	file >> m_learningRate;
	file >> m_miniBatchSize;
	file >> m_bgIsRendered;
	file >> m_layersBgAreRendered;
	
	file >> pos.x >> pos.y;
	file >> size.x >> size.y;

	file >> bgColorRed >> bgColorGreen >> bgColorBlue >> bgColorAlfa;
	file >> layersBgColorRed >> layersBgColorGreen >> layersBgColorBlue >> layersBgColorAlfa;
	file >> desiredOutputBgColorRed >> desiredOutputBgColorGreen >> desiredOutputBgColorBlue >> desiredOutputBgColorAlfa;
	file >> baseNeuronsColorRed >> baseNeuronsColorGreen >> baseNeuronsColorBlue >> baseNeuronsColorAlfa;

	file >> layersCount;

	bgColor = sf::Color(
		bgColorRed, 
		bgColorGreen, 
		bgColorBlue, 
		bgColorAlfa
	);
	layersBgColor = sf::Color(
		layersBgColorRed, 
		layersBgColorGreen, 
		layersBgColorBlue, 
		layersBgColorAlfa
	);
	desiredOutputBgColor = sf::Color(
		desiredOutputBgColorRed,
		desiredOutputBgColorGreen,
		desiredOutputBgColorBlue,
		desiredOutputBgColorAlfa
	);
	baseNeuronsColor = sf::Color(
		baseNeuronsColorRed,
		baseNeuronsColorGreen,
		baseNeuronsColorBlue,
		baseNeuronsColorAlfa
	);

	while (layersCount--)
	{
		unsigned a;
		file >> a;
		topology.push_back(a);
	}

	initLayers(topology, pos, size, layersBgColor, baseNeuronsColor);
	initSynapses(topology);
	initBg(pos, size, bgColor);
	initDesiredOutputsRenderer(
		topology, 
		pos, 
		size, 
		desiredOutputBgColor, 
		baseNeuronsColor
	);

	// load biases:
	for (int i = 1; i < topology.size(); i++)
	{
		for (int j = 0; j < topology[i]; j++)
		{
			Scalar a;
			file >> a;
			m_layers[i]->setBias(j, a);
		}
	}

	// load weights:
	for (int i = 1; i < topology.size(); i++)
	{
		for (int n = 0; n < topology[i]; n++)
		{
			for (int p = 0; p < topology[i - 1]; p++)
			{
				Scalar a;
				file >> a;
				m_synapses[i - 1].setWeight(n, p, a);
			}
		}
	}

	file.close();
}

const std::vector<Scalar>& NeuralNet::getOutput() const
{
	return m_layers.back()->getOutput();
}

const std::vector<Scalar>& NeuralNet::predict(const std::vector<Scalar>& input)
{
	propagateForward(input);
	return getOutput();
}

void NeuralNet::trainingStep(
	const std::vector<Scalar>& input, 
	const std::vector<Scalar>& desiredOutput)
{
	propagateForward(input);
	calcDerivatives();	
	m_layers.back()->calcErrors(desiredOutput);
	propagateErrorsBack();
	updateGradients();

	if ((m_trainingStep + 1U) % m_miniBatchSize == 0)
	{
		updateWeights();
		updateBiases();
		resetGradients();
	}

	m_trainingStep++;
}

void NeuralNet::updateRendering(const std::vector<Scalar>& desiredOutput)
{
	for (auto& layer : m_layers)
	{
		layer->updateRendering();
	}

	for (auto& synapsesMatrix : m_synapses)
	{
		synapsesMatrix.updateRendering();
	}

	m_desiredOutputsRenderer->setDesiredOutput(desiredOutput);
}

void NeuralNet::render(sf::RenderTarget& target)
{
	if (m_bgIsRendered)
	{
		target.draw(m_bg);
	}

	for (const auto& layer : m_layers)
	{
		layer->render(target, m_layersBgAreRendered);
	}

	m_desiredOutputsRenderer->render(target, m_layersBgAreRendered);

	for (const auto& matrixSynapse : m_synapses)
	{
		matrixSynapse.render(target);
	}
}

void NeuralNet::initLayers(
	const std::vector<unsigned>& topology,
	const sf::Vector2f& pos,
	const sf::Vector2f& size,
	const sf::Color& layersBgColor,
	const sf::Color& baseNeuronsColor)
{
	float neuronDiameter = calcNeuronDiameter(topology, size.y);
	float spaceBetweenLayers = calcSpaceBetweenLayers(topology, size);

	m_layers.clear();

	for (int i = 0; i < topology.size(); i++)
	{
		if (i == 0) // input layer
		{
			m_layers.emplace_back(
				std::make_unique<InputLayer>(
					topology[0],
					pos,
					layersBgColor,
					(topology[0] - getBiggestNonInputLayerSize(topology)) / 2U,
					getBiggestNonInputLayerSize(topology),
					neuronDiameter,
					neuronDiameter,
					baseNeuronsColor
				)
			);
		}
		else if (i == topology.size() - 1) // output layer
		{
			m_layers.emplace_back(
				std::make_unique<OutputLayer>(
					topology.back(),
					sf::Vector2f(
						pos.x + i * (neuronDiameter + spaceBetweenLayers),
						pos.y
					),
					layersBgColor,
					neuronDiameter,
					neuronDiameter,
					baseNeuronsColor
				)
			);
		}
		else // hidden layer
		{
			m_layers.emplace_back(
				std::make_unique<HiddenLayer>(
					topology[i],
					sf::Vector2f(
						pos.x + i * (neuronDiameter + spaceBetweenLayers),
						pos.y
					),
					layersBgColor,
					neuronDiameter,
					neuronDiameter,
					baseNeuronsColor
				)
			);
		}
	}

	alignNonInputLayersVertically(size);
}

void NeuralNet::initSynapses(const std::vector<unsigned>& topology)
{
	m_synapses.clear();

	for (int i = 0; i < topology.size() - 1; i++)
	{
		if (i == 0)
		{
			m_synapses.emplace_back(
				SynapsesMatrix(
					m_layers[1]->getNeurons(),
					m_layers[0]->getSize(),
					m_layers[0]->getRenderedInputsCircles(),
					m_layers[0]->getIdxOfFirstRenderedNetInput(),
					m_layers[0]->getNumberOfRenderedNetInputs()
				)
			);
		}
		else
		{
			m_synapses.emplace_back(
				SynapsesMatrix(
					m_layers[i + 1]->getNeurons(),
					m_layers[i]->getNeurons()
				)
			);
		}
	}
}

void NeuralNet::initBg(
	const sf::Vector2f& pos, 
	const sf::Vector2f& size, 
	const sf::Color& bgColor)
{
	m_bg.setPosition(pos);
	m_bg.setSize(size);
	m_bg.setFillColor(bgColor);
}

void NeuralNet::initDesiredOutputsRenderer(
	const std::vector<unsigned>& topology, 
	const sf::Vector2f& pos, 
	const sf::Vector2f& size,
	const sf::Color& desiredOutputsBgColor,
	const sf::Color& baseColorOfDesiredOutputsCircles)
{
	float neuronDiameter = calcNeuronDiameter(topology, size.y);

	sf::Vector2f desiredOutputsRendererPos = sf::Vector2f(
		m_layers.back()->getPos().x + m_layers.back()->getRenderingSize().x,
		m_layers.back()->getPos().y
	);

	m_desiredOutputsRenderer = std::make_unique<DesiredOutputsRenderer>(
		DesiredOutputsRenderer(
			topology.back(),
			desiredOutputsRendererPos,
			desiredOutputsBgColor,
			neuronDiameter,
			neuronDiameter,
			baseColorOfDesiredOutputsCircles
		)
	);
}

float NeuralNet::calcNeuronDiameter(const std::vector<unsigned>& topology, float netHeight)
{
	unsigned howManyDiametersFitInBiggestNonInputLayer = 2U * getBiggestNonInputLayerSize(topology) - 1U;
	return netHeight / howManyDiametersFitInBiggestNonInputLayer;
}

unsigned NeuralNet::getBiggestNonInputLayerSize(const std::vector<unsigned>& topology)
{
	unsigned result = 0U;

	for (int i = 1; i < topology.size(); i++)
	{
		result = std::max(result, topology[i]);
	}

	return result;
}

float NeuralNet::calcSpaceBetweenLayers(
	const std::vector<unsigned>& topology, 
	const sf::Vector2f& size)
{
	float neuronDiameter = calcNeuronDiameter(topology, size.y);
	return (size.x - (topology.size() + 1) * neuronDiameter) / (topology.size() - 1);
}

void NeuralNet::alignNonInputLayersVertically(const sf::Vector2f& size)
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		m_layers[i]->moveVertically(
			(size.y - m_layers[i]->getRenderingSize().y) / 2.0f
		);
	}
}

void NeuralNet::propagateForward(const std::vector<Scalar>& input)
{
	m_layers[0]->setInput(input);

	for (int i = 1; i < m_layers.size(); i++)
	{
		if (i == 1)
		{
			m_layers[i]->propagateForward(
				m_layers[0]->getInput(),
				m_synapses[i - 1]
			);
		}
		else
		{
			m_layers[i]->propagateForward(
				*m_layers[i - 1].get(),
				m_synapses[i - 1]
			);
		}
	}
}

void NeuralNet::calcDerivatives()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		m_layers[i]->calcDerivatives();
	}
}

void NeuralNet::propagateErrorsBack()
{
	for (int i = m_layers.size() - 2; i > 0; i--)
	{
		m_layers[i]->propagateErrorsBack(
			*m_layers[i + 1].get(), 
			m_synapses[i]
		);
	}
}

void NeuralNet::updateGradients()
{
	updateWeightsGradients();
	updateBiasesGradients();
}

void NeuralNet::updateWeightsGradients()
{
	for (int i = 0; i < m_synapses.size(); i++)
	{
		if (i == 0)
		{
			m_synapses[i].updateWeightsGradients(
				m_layers[i]->getInput(),
				m_layers[i + 1]->getNeurons()
			);
		}
		else
		{
			m_synapses[i].updateWeightsGradients(
				m_layers[i]->getNeurons(),
				m_layers[i + 1]->getNeurons()
			);
		}
	}
}

void NeuralNet::updateBiasesGradients()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		m_layers[i]->updateBiasesGradients();
	}
}

Scalar NeuralNet::getBiggestAbsValOfWeight() const
{
	Scalar result = 0.0;

	for (const auto& synapsesMatrix : m_synapses)
	{
		for (const auto& it1 : synapsesMatrix.getSynapsesMatrix())
		{
			for (const auto& it2 : it1)
			{
				// I'm not sure if I can trust std::max(result, it2.getWeight())
				if (result < std::abs(it2.getWeight()))
				{
					result = std::abs(it2.getWeight());
				}
			}
		}
	}

	return result;
}

bool NeuralNet::isBgRendered() const
{
	return m_bgIsRendered;
}

void NeuralNet::hideBg()
{
	m_bgIsRendered = false;
}

void NeuralNet::showBg()
{
	m_bgIsRendered = true;
}

bool NeuralNet::areLayersBgRendered() const
{
	return m_layersBgAreRendered;
}

void NeuralNet::hideLayersBg()
{
	m_layersBgAreRendered = false;
}

void NeuralNet::showLayersBg()
{
	m_layersBgAreRendered = true;
}

void NeuralNet::updateWeights()
{
	for (auto& synapsesMatrix : m_synapses)
	{
		for (int n = 0; n < synapsesMatrix.getDimensions().first; n++)
		{
			for (int p = 0; p < synapsesMatrix.getDimensions().second; p++)
			{
				Scalar change =
					m_learningRate *
					synapsesMatrix.getSynapsesMatrix()[n][p].getGradient() /
					m_miniBatchSize;

				synapsesMatrix.setWeight(
					n,
					p,
					synapsesMatrix.getWeight(n, p) - change
				);
			}
		}
	}
}

void NeuralNet::updateBiases()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		for (int j = 0; j < m_layers[i]->getSize(); j++)
		{
			Scalar change = 
				m_learningRate * 
				m_layers[i]->getNeurons()[j].getBiasGradient() / 
				m_miniBatchSize;

			m_layers[i]->setBias(
				j,
				m_layers[i]->getBias(j) - change
			);
		}
	}
}

void NeuralNet::resetGradients()
{
	resetWeightsGradients();
	resetBiasesGradients();
}

void NeuralNet::resetWeightsGradients()
{
	for (auto& synapseMatrix : m_synapses)
	{
		synapseMatrix.resetGradients();
	}
}

void NeuralNet::resetBiasesGradients()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		m_layers[i]->resetBiasesGradients();
	}
}

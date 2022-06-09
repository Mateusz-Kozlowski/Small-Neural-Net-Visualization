#include "NeuralNet.h"

NeuralNet::NeuralNet(
	const std::vector<unsigned>& topology,
	const Scalar& learningRate, 
	unsigned miniBatchSize,
	const sf::Vector2f& pos,
	const sf::Vector2f& size,
	const sf::Color& bgColor)
	: m_trainingStep(0U),
	  m_learningRate(learningRate), 
	  m_miniBatchSize(miniBatchSize)
{
	initLayers(pos, topology);
	initSynapses(topology);
	initBg(pos, size, bgColor);
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

	file << m_layers.size() << '\n';
	for (int i = 0; i < m_layers.size(); i++)
	{
		file << m_layers[i]->getSize() << ' ';
	}

	for (int i=1; i<m_layers.size(); i++)
	{
		//std::cout << "saving " << i << " layer biases:\n";
		for (int j=0; j<m_layers[i]->getSize(); j++)
		{
			//std::cout << "j=" << j << '\n';

			auto temp = m_layers[i]->getNeurons()[j];
			
			int xd = 0;
			xd++;

			file << m_layers[i]->getNeurons()[j].getBias() << ' ';

			xd--;
		}
	}
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

	//std::cout << "saving completed\n";
}

void NeuralNet::load(const std::string& path)
{
	//std::cerr << "LOL KURWA\n";

	std::ifstream file(path);

	if (!file.is_open())
	{
		exit(-13);
	}

	unsigned layersCount;
	std::vector<unsigned> topology;

	file >> layersCount;
	if (layersCount != m_layers.size())
	{
		std::cerr << layersCount << ' ' << m_layers.size() << '\n';
		std::cerr << "LOL KURWA\n";
		exit(-11);
	}

	while (layersCount--)
	{
		unsigned a;
		file >> a;
		topology.push_back(a);
	}

	for (int i = 0; i < 3; i++)
	{
		if (topology[i] != m_layers[i]->getSize())
		{
			std::cerr << "LOL KURWA12\n";
			exit(-12);
		}
	}

	for (int i = 1; i < topology.size(); i++)
	{
		for (int j = 0; j < topology[i]; j++)
		{
			Scalar a;
			file >> a;
			m_layers[i]->setBias(j, a);
		}
	}

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
	
	//saveGradients();
	//saveWeightsAndBiases();

	if ((m_trainingStep + 1U) % m_miniBatchSize == 0)
	{
		updateWeights();
		updateBiases();
		resetGradients();
	}

	m_trainingStep++;
}

void NeuralNet::updateRendering()
{
	for (auto& layer : m_layers)
	{
		layer->updateRendering();
	}
}

void NeuralNet::render(sf::RenderTarget& target)
{
	target.draw(m_bg);

	for (const auto& layer : m_layers)
	{
		layer->render(target);
	}
}

void NeuralNet::initLayers(
	const sf::Vector2f& pos, 
	const std::vector<unsigned>& topology)
{
	for (int i = 0; i < topology.size(); i++)
	{
		if (i == 0) // input layer
		{
			m_layers.emplace_back(
				std::make_unique<InputLayer>(
					topology[i],
					pos,
					sf::Color::Magenta,
					(topology[0] - getBiggestNonInputLayerSize(topology)) / 2U,
					getBiggestNonInputLayerSize(topology),
					32.0f, // TODO: unhardcode these
					32.0f
				)
			);
		}
		else if (i == topology.size() - 1) // output layer
		{
			m_layers.emplace_back(
				std::make_unique<OutputLayer>(
					topology[i],
					sf::Vector2f(
						pos.x + i * 256.0f,
						pos.y
					),
					sf::Color::Magenta,
					32.0f,
					32.0f
				)
			);
		}
		else // hidden layer
		{
			m_layers.emplace_back(
				std::make_unique<HiddenLayer>(
					topology[i],
					sf::Vector2f(
						pos.x + i * 256.0f,
						pos.y
					),
					sf::Color::Magenta,
					32.0f,
					32.0f
				)
			);
		}
	}
}

void NeuralNet::initSynapses(const std::vector<unsigned>& topology)
{
	for (int i = 0; i < topology.size() - 1; i++)
	{
		m_synapses.emplace_back(
			SynapsesMatrix(
				topology[i + 1],
				topology[i]
			)
		);
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

unsigned NeuralNet::getBiggestNonInputLayerSize(const std::vector<unsigned>& topology)
{
	unsigned result = 0U;

	for (int i = 1; i < topology.size(); i++)
	{
		result = std::max(result, topology[i]);
	}

	return result;
}

const std::vector<Scalar>& NeuralNet::getOutput() const
{
	return m_layers.back()->getOutput();
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

			//std::cout << "from net:\n";
			//std::cout << m_synapses[0].getSynapsesMatrix()[0][0].getGradient() << '\n';
		}
		else
		{
			m_synapses[i].updateWeightsGradients(
				m_layers[i]->getNeurons(),
				m_layers[i + 1]->getNeurons()
			);
		}
	}

	//std::cout << "from net:\n";
	//std::cout << m_synapses[0].getSynapsesMatrix()[0][0].getGradient() << '\n';
}

void NeuralNet::updateBiasesGradients()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		m_layers[i]->updateBiasesGradients();
	}
}

void NeuralNet::saveGradients()
{
	std::string path = "gradienciki po 1ST minibaczu.ini";

	std::ofstream gradienciki(path);

	if (!gradienciki.is_open())
	{
		std::cerr << "CANNOT OPEN: " << path << '\n';
		exit(-13);
	}

	for (int i = 1; i < m_layers.size(); i++)
	{
		for (int j = 0; j < m_layers[i]->getSize(); j++)
		{
			gradienciki << m_layers[i]->getNeurons()[j].getBiasGradient() << '\n';
		}
	}

	gradienciki << "Weights gradients:\n";

	int idx = 0;
	for (const auto& synapsesMatrix : m_synapses)
	{
		for (const auto& it1 : synapsesMatrix.getSynapsesMatrix())
		{
			for (const auto& it2 : it1)
			{
				gradienciki << it2.getGradient() << '\n';
				idx++;
			}
		}
	}

	gradienciki.close();
	//exit(7);
}

void NeuralNet::saveWeightsAndBiases()
{
	std::string path = "w&b po 1ST minibaczu.ini";

	std::ofstream wb(path);

	if (!wb.is_open())
	{
		std::cerr << "CANNOT OPEN: " << path << '\n';
		exit(-13);
	}

	for (int i = 1; i < m_layers.size(); i++)
	{
		for (int j = 0; j < m_layers[i]->getSize(); j++)
		{
			wb << m_layers[i]->getNeurons()[j].getBias() << '\n';
		}
	}

	wb << "Weights:\n";

	int idx = 0;
	for (const auto& synapsesMatrix : m_synapses)
	{
		for (const auto& it1 : synapsesMatrix.getSynapsesMatrix())
		{
			for (const auto& it2 : it1)
			{
				wb << it2.getWeight() << '\n';
				idx++;
			}
		}
	}

	wb.close();
	//exit(7);
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

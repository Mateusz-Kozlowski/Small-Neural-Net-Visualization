#include "App.h"

App::App()
	: m_trainingDataIdx(-1), m_epochIdx(1U), m_learingPaused(false)
{
	initWindow();
	loadData();
	initNet();
	initDataPointRenderer();
}

void App::run()
{
	std::cout << "App is now running:\n";

	while (m_window.isOpen())
	{
		update();
		render();
	}
}

void App::initWindow()
{
	std::string path = "config/graphics.ini";

	std::ifstream file(path);

	if (!file.is_open())
	{
		std::cerr << "CANNOT OPEN: " << path << '\n';
		exit(-17);
	}

	std::string title;
	unsigned width, height;
	unsigned frameRateLimit;
	bool verticalSyncEnabled;
	sf::Vector2i position;
	bool keyRepeatEnabled;

	getline(file, title);
	file >> width >> height;
	file >> frameRateLimit;
	file >> verticalSyncEnabled;
	file >> position.x >> position.y;
	file >> keyRepeatEnabled;

	file.close();

	std::cout << "window info:\n";
	std::cout << "title: " << title << '\n';
	std::cout << "width: " << width << '\n';
	std::cout << "height: " << height << '\n';
	std::cout << "FRL: " << frameRateLimit << '\n';
	std::cout << "verticalSyncEnabled: " << verticalSyncEnabled << '\n';
	std::cout << "pos: " << position.x << ' ' << position.y << '\n';

	m_window.setTitle(title);
	m_window.create(sf::VideoMode(width, height), title);
	m_window.setFramerateLimit(frameRateLimit);
	m_window.setVerticalSyncEnabled(verticalSyncEnabled);
	m_window.setPosition(position);
	m_window.setKeyRepeatEnabled(keyRepeatEnabled);
}

void App::loadData()
{
	std::cout << "Loading data...\n";

	m_trainInputs = Utils::MNISTdataLoader::readMnistImages(
		"handwritten digits data//train-images-idx3-ubyte"
	);
	auto trainLabels = Utils::MNISTdataLoader::readMnistLabels(
		"handwritten digits data//train-labels-idx1-ubyte"
	);

	m_testInputs = Utils::MNISTdataLoader::readMnistImages(
		"handwritten digits data//t10k-images-idx3-ubyte"
	);
	auto testLabels = Utils::MNISTdataLoader::readMnistLabels(
		"handwritten digits data//t10k-labels-idx1-ubyte"
	);

	std::cout << "\nThis is how 2 first data points look like:\n";

	Utils::MNISTdataLoader::showData(m_trainInputs, trainLabels, { 0U, 1U });

	Utils::MNISTdataLoader::preprocess(m_trainInputs);
	Utils::MNISTdataLoader::preprocess(m_testInputs);

	m_trainLabels = Utils::MNISTdataLoader::reorganizeLabels(trainLabels);
	m_testLabels = Utils::MNISTdataLoader::reorganizeLabels(testLabels);
}

void App::initNet()
{
	std::string path = "config/net.ini";

	std::ifstream file(path);

	if (!file.is_open())
	{
		std::cerr << "CANNOT OPEN: " << path << '\n';
		exit(-17);
	}

	unsigned layersCount;
	std::vector<unsigned> topology;
	float learningRate;
	unsigned miniBatchSize;
	sf::Vector2f pos;
	float width;
	unsigned red, green, blue, alfa;

	file >> layersCount;
	topology.resize(layersCount);
	for (int i = 0; i < layersCount; i++)
	{
		file >> topology[i];
	}

	file >> learningRate;
	file >> miniBatchSize;
	file >> pos.x >> pos.y;
	file >> width;
	file >> red >> green >> blue >> alfa;

	file.close();

	sf::Vector2f size(
		width,
		m_window.getSize().y - 2.0f * pos.y
	);

	m_net = std::make_unique<NeuralNet>(
		NeuralNet(
			topology,
			learningRate,
			miniBatchSize,
			pos,
			size,
			sf::Color(red, green, blue, alfa)
		)
	);

	m_net->save("brand new and dumb net.ini");
}

void App::initDataPointRenderer()
{
	float size = 256.0f;
	float dataPointRendererLeftMargin =
		0.5f * (
			m_window.getSize().x
			- m_net->getPos().x
			- m_net->getSize().x
			- size
	);

	sf::Vector2f pos(
		m_net->getPos().x + m_net->getSize().x + dataPointRendererLeftMargin,
		m_net->getPos().y
	);

	m_dataPointRenderer = std::make_unique<DataPointRenderer>(
		DataPointRenderer(
			pos,
			sf::Vector2f(
				size,
				size
			),
			sf::Color::Magenta,
			4.0f,
			m_testInputs[0].size()
		)
	);
}

void App::update()
{
	updateEvents();
	updateLearningProcess();
	updateRendering();
}

void App::render()
{
	m_window.clear();
	
	m_net->render(m_window);
	m_dataPointRenderer->render(m_window);
	
	m_window.display();
}

void App::updateEvents()
{
	sf::Event event;

	while (m_window.pollEvent(event))
	{
		if (event.type == sf::Event::Closed)
		{
			m_window.close();
		}
		if (event.type == sf::Event::KeyPressed)
		{
			if (event.key.code == sf::Keyboard::N)
			{
				if (m_net->isBgRendered())
				{
					m_net->hideBg();
				}
				else
				{
					m_net->showBg();
				}
			}
			if (event.key.code == sf::Keyboard::L)
			{
				if (m_net->areLayersBgRendered())
				{
					m_net->hideLayersBg();
				}
				else
				{
					m_net->showLayersBg();
				}
			}
			if (event.key.code == sf::Keyboard::Space)
			{
				m_learingPaused = !m_learingPaused;

				for (const auto& it : m_trainLabels[m_trainingDataIdx])
				{
					std::cout << it << ' ';
				}
				std::cout << '\n';

				Utils::MNISTdataLoader::showData(
					m_trainInputs,
					m_trainLabels,
					{
						m_trainingDataIdx,
						m_trainingDataIdx
					}
				);

				auto output = m_net->getOutput();

				std::cout << "output:\n";
				for (int j = 0; j < output.size(); j++)
				{
					std::cout << "j: " << output[j] << '\n';
				}
			}
		}
	}
}

void App::updateLearningProcess()
{
	if (m_learingPaused)
	{
		return;
	}
	
	m_trainingDataIdx++;

	if (m_trainingDataIdx % 10'000 == 0 && m_trainingDataIdx != 0)
	{
		std::cout << m_trainingDataIdx << '\n';
	}

	if (m_trainingDataIdx == m_trainInputs.size())
	{
		m_trainingDataIdx = 0;

		m_net->save("the newest version of net.ini");
		
		std::cout << "Accuracy after " << m_epochIdx << " epoch: " << 100.0f * Utils::validateClassification(
			m_testInputs,
			m_testLabels,
			*m_net.get()
		) << "%\n";

		m_epochIdx++;

		Utils::randomShuffle(m_trainInputs, m_trainLabels);
	}

	m_net->trainingStep(
		m_trainInputs[m_trainingDataIdx],
		m_trainLabels[m_trainingDataIdx]
	);
}

void App::updateRendering()
{
	m_net->updateRendering(m_trainLabels[m_trainingDataIdx]);
	m_dataPointRenderer->updateRendering(m_trainInputs[m_trainingDataIdx]);
}

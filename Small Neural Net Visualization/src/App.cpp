#include "App.h"

App::App()
{
	initWindow();
	loadData();
}

void App::run()
{
	NeuralNet net({ 784U, 32U, 16U, 10U }, 1.0, 32U);

	unsigned epochsCount = 1U;
	std::cout << "Learning epochs count: ";
	std::cin >> epochsCount;

	for (int e = 1; e <= epochsCount; e++)
	{
		for (int i = 0; i < m_trainInputs.size(); i++)
		{
			net.trainingStep(m_trainInputs[i], m_trainLabels[i]);
		}

		std::cout << "Accuracy after " << e << " epoch: " << Utils::validateClassification(
			m_testInputs,
			m_testLabels,
			net
		) << '\n';

		Utils::randomShuffle(m_trainInputs, m_trainLabels);
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

	file >> title;
	file >> width >> height;
	file >> frameRateLimit;
	file >> verticalSyncEnabled;
	file >> position.x >> position.y;

	file.close();

	m_window.create(sf::VideoMode(width, height), title);
	m_window.setFramerateLimit(frameRateLimit);
	m_window.setVerticalSyncEnabled(verticalSyncEnabled);
	m_window.setPosition(position);
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

	Utils::MNISTdataLoader::preprocess(m_trainInputs);
	Utils::MNISTdataLoader::preprocess(m_testInputs);

	m_trainLabels = Utils::MNISTdataLoader::reorganizeLabels(trainLabels);
	m_testLabels = Utils::MNISTdataLoader::reorganizeLabels(testLabels);

	std::cout << "\nThis is how 2 first data points look like:\n";

	Utils::MNISTdataLoader::showData(m_trainInputs, trainLabels, { 0U, 1U });
}

void App::update()
{

}

void App::updateDt()
{

}

void App::render()
{

}

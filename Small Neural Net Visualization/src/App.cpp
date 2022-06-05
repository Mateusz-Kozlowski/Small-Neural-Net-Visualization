#include "App.h"

App::App()
{
	initWindow();
	loadData();
	initNet();
}

void App::run()
{
	NeuralNet net({ 784U, 16U, 12U, 10U }, 1.0, 32U);

	unsigned epochsCount = 1U;
	std::cout << "Learning epochs count: ";
	//std::cin >> epochsCount;
	epochsCount = 9;
	std::cout << '\n';

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

#pragma once

#include "Utils.h"
#include "DataPointRenderer.h"

class App
{
public:
	App();

	void run();

private:
	void initWindow();
	void loadData();
	void initNet();
	void initDataPointRenderer();

	void update();
	void render();

	void updateEvents();
	void updateLearningProcess();
	void updateRendering();

	std::unique_ptr<NeuralNet> m_net;
	sf::RenderWindow m_window;

	unsigned m_trainingDataIdx;
	unsigned m_epochIdx;

	std::vector<std::vector<Scalar>> m_trainInputs;
	std::vector<std::vector<Scalar>> m_trainLabels;

	std::vector<std::vector<Scalar>> m_testInputs;
	std::vector<std::vector<Scalar>> m_testLabels;

	std::unique_ptr<DataPointRenderer> m_dataPointRenderer;
};

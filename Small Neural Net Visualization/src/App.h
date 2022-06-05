#pragma once

#include "Utils.h"

class App
{
public:
	App();

	void run();

private:
	void initWindow();
	void loadData();
	void initNet();

	void update();
	void updateDt();
	void render();

	std::vector<std::vector<Scalar>> m_trainInputs;
	std::vector<std::vector<Scalar>> m_trainLabels;

	std::vector<std::vector<Scalar>> m_testInputs;
	std::vector<std::vector<Scalar>> m_testLabels;
};

#pragma once

#include "Utils.h"

#include "SFML/Graphics.hpp"

class App
{
public:
	App();

	void run();

private:
	void initWindow();
	void loadData();

	void update();
	void updateDt();
	void render();

	sf::RenderWindow m_window;

	std::vector<std::vector<Scalar>> m_trainInputs;
	std::vector<std::vector<Scalar>> m_trainLabels;

	std::vector<std::vector<Scalar>> m_testInputs;
	std::vector<std::vector<Scalar>> m_testLabels;
};

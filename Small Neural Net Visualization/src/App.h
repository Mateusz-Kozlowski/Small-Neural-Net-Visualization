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
};

#include "RandomEngine.h"

bool RandomEngine::s_initialized = false;
std::default_random_engine RandomEngine::s_eng;

// public methods:

int RandomEngine::getIntInRange(int first, int last)
{
	return getNumberInRange<std::uniform_int_distribution<int>>(first, last);
}

Scalar RandomEngine::getScalarInRange(Scalar first, Scalar last)
{
	return getNumberInRange<std::uniform_real_distribution<Scalar>>(first, last);
}

// private methods:

void RandomEngine::init()
{
	std::random_device rd;
	s_eng = std::default_random_engine(rd());
}

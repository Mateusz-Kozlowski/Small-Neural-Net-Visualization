#include "InputLayer.h"

InputLayer::InputLayer(unsigned size) : Layer(size)
{
	m_input.resize(size);
}

void InputLayer::setInput(const std::vector<Scalar>& input)
{
	m_input = input;
}

const std::vector<Scalar>& InputLayer::getInput() const
{
	return m_input;
}

const Scalar& InputLayer::getVal(unsigned idx) const
{
	return m_input[idx];
}

unsigned InputLayer::getSize() const
{
	return m_input.size();
}

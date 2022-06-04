#pragma once

#include "Layer.h"

class InputLayer : public Layer
{
public:
	InputLayer(unsigned size);

	virtual void setInput(const std::vector<Scalar>& input) override;

	virtual const std::vector<Scalar>& getInput() const override;

	virtual const Scalar& getVal(unsigned idx) const override;
	virtual unsigned getSize() const override;

private:
	std::vector<Scalar> m_input;
};

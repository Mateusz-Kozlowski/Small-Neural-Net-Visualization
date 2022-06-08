#pragma once

#include "SynapsesMatrix.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

#include <memory>
#include <iostream>
#include <fstream>

class NeuralNet
{
public:
	NeuralNet(
		const std::vector<unsigned>& topology,
		const Scalar& learningRate,
		unsigned miniBatchSize,
		const sf::Vector2f& pos,
		const sf::Vector2f& size,
		const sf::Color& bgColor
	);

	void save(const std::string& path);
	void load(const std::string& path);

	const std::vector<Scalar>& predict(const std::vector<Scalar>& input);
	
	void trainingStep(
		const std::vector<Scalar>& input,
		const std::vector<Scalar>& desiredOutput
	);

	void updateRendering();
	void render(sf::RenderTarget& target);

	void saveWeightsAndBiases();

private:
	void initVals(const std::vector<unsigned>& topology);
	void initSynapses(const std::vector<unsigned>& topology);
	void initBg(
		const sf::Vector2f& pos,
		const sf::Vector2f& size,
		const sf::Color& bgColor
	);

	const std::vector<Scalar>& getOutput() const;

	void propagateForward(const std::vector<Scalar>& input);

	void calcDerivatives();
	void propagateErrorsBack();
	
	void updateGradients();
	void updateWeightsGradients();
	void updateBiasesGradients();

	void updateWeights();
	void updateBiases();

	void resetGradients();
	void resetWeightsGradients();
	void resetBiasesGradients();

	void saveGradients();

	unsigned m_trainingStep;

	Scalar m_learningRate;
	unsigned m_miniBatchSize;
	
	std::vector<std::unique_ptr<Layer>> m_layers;
	std::vector<SynapsesMatrix> m_synapses;

	sf::RectangleShape m_bg;
};

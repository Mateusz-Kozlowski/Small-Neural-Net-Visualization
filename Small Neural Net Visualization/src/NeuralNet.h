#pragma once

#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

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

	const sf::Vector2f& getPos() const;
	const sf::Vector2f& getSize() const;

	void save(const std::string& path);
	void load(const std::string& path);

	const std::vector<Scalar>& getOutput() const;
	const std::vector<Scalar>& predict(const std::vector<Scalar>& input);
	
	void trainingStep(
		const std::vector<Scalar>& input,
		const std::vector<Scalar>& desiredOutput
	);

	void updateRendering();
	void render(sf::RenderTarget& target);

	void saveWeightsAndBiases();

	bool isBgRendered() const;
	void hideBg();
	void showBg();

	bool areLayersBgRendered() const;
	void hideLayersBg();
	void showLayersBg();

private:
	void initLayers(
		const std::vector<unsigned>& topology,
		const sf::Vector2f& pos, 
		const sf::Vector2f& size
	);
	void initSynapses(const std::vector<unsigned>& topology);
	void initBg(
		const sf::Vector2f& pos,
		const sf::Vector2f& size,
		const sf::Color& bgColor
	);

	static float calcNeuronDiameter(
		const std::vector<unsigned>& topology,
		float netHeight
	);
	static unsigned getBiggestNonInputLayerSize(const std::vector<unsigned>& topology);
	static float calcSpaceBetweenLayers(
		const std::vector<unsigned>& topology,
		const sf::Vector2f& size
	);

	void alignNonInputLayersVertically(const sf::Vector2f& size);

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

	Scalar getBiggestAbsValOfWeight() const;

	unsigned m_trainingStep;

	Scalar m_learningRate;
	unsigned m_miniBatchSize;
	
	std::vector<std::unique_ptr<Layer>> m_layers;
	std::vector<SynapsesMatrix> m_synapses;

	bool m_bgIsRendered;
	bool m_layersbgAreRendered;

	sf::RectangleShape m_bg;
};

#include "NeuralNet.h"

NeuralNet::NeuralNet(
	const std::vector<unsigned>& topology,
	const Scalar& learningRate, 
	unsigned miniBatchSize)
	: m_trainingStep(0U), 
	  m_learningRate(learningRate), 
	  m_miniBatchSize(miniBatchSize)
{
	initVals(topology);
	initSynapses(topology);
}

const std::vector<Scalar>& NeuralNet::predict(const std::vector<Scalar>& input)
{
	propagateForward(input);
	return getOutput();
}

void NeuralNet::trainingStep(
	const std::vector<Scalar>& input, 
	const std::vector<Scalar>& desiredOutput)
{
	propagateForward(input);
	calcDerivatives();	
	m_layers.back()->calcErrors(desiredOutput);
	propagateErrorsBack();
	updateGradients();

	if ((m_trainingStep + 1U) % m_miniBatchSize == 0)
	{
		updateWeights();
		updateBiases();
		resetGradients();
	}

	m_trainingStep++;
}

void NeuralNet::initVals(const std::vector<unsigned>& topology)
{
	for (int i = 0; i < topology.size(); i++)
	{
		if (i == 0) // input layer
		{
			m_layers.emplace_back(
				std::make_unique<InputLayer>(topology[i])
			);
		}
		else if (i == topology.size() - 1) // output layer
		{
			m_layers.emplace_back(
				std::make_unique<OutputLayer>(topology[i])
			);
		}
		else // hidden layer
		{
			m_layers.emplace_back(
				std::make_unique<HiddenLayer>(topology[i])
			);
		}
	}
}

void NeuralNet::initSynapses(const std::vector<unsigned>& topology)
{
	for (int i = 0; i < topology.size() - 1; i++)
	{
		m_synapses.emplace_back(
			SynapsesMatrix(
				topology[i + 1],
				topology[i]
			)
		);
	}
}

const std::vector<Scalar>& NeuralNet::getOutput() const
{
	return m_layers.back()->getOutput();
}

void NeuralNet::propagateForward(const std::vector<Scalar>& input)
{
	m_layers[0]->setInput(input);

	for (int i = 1; i < m_layers.size(); i++)
	{
		if (i == 1)
		{
			m_layers[i]->propagateForward(
				m_layers[0]->getInput(),
				m_synapses[i - 1]
			);
		}
		else
		{
			m_layers[i]->propagateForward(
				*m_layers[i - 1].get(),
				m_synapses[i - 1]
			);
		}
	}
}

void NeuralNet::calcDerivatives()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		m_layers[i]->calcDerivatives();
	}
}

void NeuralNet::propagateErrorsBack()
{
	for (int i = m_layers.size() - 2; i > 0; i--)
	{
		m_layers[i]->propagateErrorsBack(
			*m_layers[i + 1].get(), 
			m_synapses[i]
		);
	}
}

void NeuralNet::updateGradients()
{
	updateWeightsGradients();
	updateBiasesGradients();
}

void NeuralNet::updateWeightsGradients()
{
	for (int i = 0; i < m_synapses.size(); i++)
	{
		if (i == 0)
		{
			m_synapses[i].updateWeightsGradients(
				m_layers[i]->getInput(),
				m_layers[i + 1]->getNeurons()
			);
		}
		else
		{
			m_synapses[i].updateWeightsGradients(
				m_layers[i]->getNeurons(),
				m_layers[i + 1]->getNeurons()
			);
		}
	}
}

void NeuralNet::updateBiasesGradients()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		m_layers[i]->updateBiasesGradients();
	}
}

void NeuralNet::updateWeights()
{
	for (auto& synapsesMatrix : m_synapses)
	{
		for (int n = 0; n < synapsesMatrix.getDimensions().first; n++)
		{
			for (int p = 0; p < synapsesMatrix.getDimensions().second; p++)
			{
				Scalar change =
					m_learningRate *
					synapsesMatrix.getSynapsesMatrix()[n][p].getGradient() /
					m_miniBatchSize;

				synapsesMatrix.setWeight(
					n,
					p,
					synapsesMatrix.getWeight(n, p) - change
				);
			}
		}
	}
}

void NeuralNet::updateBiases()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		for (int j = 0; j < m_layers[i]->getSize(); j++)
		{
			Scalar change = 
				m_learningRate * 
				m_layers[i]->getNeurons()[j].getBiasGradient() / 
				m_miniBatchSize;

			m_layers[i]->setBias(
				j,
				m_layers[i]->getBias(j) - change
			);
		}
	}
}

void NeuralNet::resetGradients()
{
	resetWeightsGradients();
	resetBiasesGradients();
}

void NeuralNet::resetWeightsGradients()
{
	for (auto& synapseMatrix : m_synapses)
	{
		synapseMatrix.resetGradients();
	}
}

void NeuralNet::resetBiasesGradients()
{
	for (int i = 1; i < m_layers.size(); i++)
	{
		m_layers[i]->resetBiasesGradients();
	}
}

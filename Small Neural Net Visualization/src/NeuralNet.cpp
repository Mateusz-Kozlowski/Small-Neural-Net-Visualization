#include "NeuralNet.h"

NeuralNet::NeuralNet(
	const std::vector<unsigned>& topology,
	const Scalar& learningRate, 
	unsigned miniBatchSize)
	: m_trainingStep(0U), m_learningRate(learningRate), m_miniBatchSize(miniBatchSize)
{
	initVals(topology);
	initWeights(topology);
	initBiases(topology);
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
	calcErrors(desiredOutput);
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
	m_val.resize(topology.size());
	m_actVal.resize(topology.size() - 1);
	m_derivatives.resize(topology.size() - 1);
	m_lossDerivativeWithRespectToActFunc.resize(topology.size() - 1);

	m_val[0].resize(topology[0]);

	for (int i = 1; i < topology.size(); i++)
	{
		m_val[i].resize(topology[i]);
		m_actVal[i - 1].resize(topology[i]);
		m_derivatives[i - 1].resize(topology[i]);
		m_lossDerivativeWithRespectToActFunc[i - 1].resize(topology[i]);
	}
}

void NeuralNet::initWeights(const std::vector<unsigned>& topology)
{
	std::random_device rd;
	std::default_random_engine eng = std::default_random_engine(rd());
	std::uniform_real_distribution<Scalar> dist(-1.0f, 1.0f);

	m_weights.resize(topology.size() - 1);
	m_weightsGradient.resize(topology.size() - 1);

	for (int i = 0; i < m_weights.size(); i++)
	{
		m_weights[i].resize(topology[i + 1]);
		m_weightsGradient[i].resize(topology[i + 1]);

		for (int n = 0; n < m_weights[i].size(); n++)
		{
			m_weights[i][n].resize(topology[i]);
			m_weightsGradient[i][n].resize(topology[i]);

			for (int p = 0; p < m_weights[i][n].size(); p++)
			{
				m_weights[i][n][p] = dist(eng);
			}
		}
	}
}

void NeuralNet::initBiases(const std::vector<unsigned>& topology)
{
	m_biases.resize(topology.size() - 1);
	m_biasesGradient.resize(topology.size() - 1);

	for (int i = 0; i < topology.size() - 1; i++)
	{
		m_biases[i].resize(topology[i + 1]);
		m_biasesGradient[i].resize(topology[i + 1]);
	}
}

const std::vector<Scalar>& NeuralNet::getOutput() const
{
	return m_actVal.back();
}

void NeuralNet::propagateForward(const std::vector<Scalar>& input)
{
	m_val[0] = input;

	for (int i = 0; i < m_weights.size(); i++)
	{
		for (int n = 0; n < m_actVal[i].size(); n++)
		{
			m_val[i + 1][n] = 0.0;

			if (i == 0)
			{
				for (int p = 0; p < input.size(); p++)
				{
					m_val[i + 1][n] += input[p] * m_weights[i][n][p];
				}
			}
			else
			{
				for (int p = 0; p < m_actVal[i - 1].size(); p++)
				{
					m_val[i + 1][n] += m_actVal[i - 1][p] * m_weights[i][n][p];
				}
			}

			Scalar biasedVal = m_val[i + 1][n] + m_biases[i][n];

			m_actVal[i][n] = 1.0 / (1.0 + exp(-biasedVal));
		}
	}
}

void NeuralNet::calcDerivatives()
{
	for (int i = 0; i < m_derivatives.size(); i++)
	{
		for (int j = 0; j < m_derivatives[i].size(); j++)
		{
			m_derivatives[i][j] = m_actVal[i][j] * (1.0 - m_actVal[i][j]);
		}
	}
}

void NeuralNet::calcErrors(const std::vector<Scalar>& desiredOutputs)
{
	for (int i = 0; i < desiredOutputs.size(); i++)
	{
		m_lossDerivativeWithRespectToActFunc.back()[i] = m_actVal.back()[i] - desiredOutputs[i];
	}
}

void NeuralNet::propagateErrorsBack()
{
	for (int i = m_lossDerivativeWithRespectToActFunc.size() - 2; i >= 0; i--)
	{
		for (int j = 0; j < m_lossDerivativeWithRespectToActFunc[i].size(); j++)
		{
			m_lossDerivativeWithRespectToActFunc[i][j] = 0.0;
			
			for (int n = 0; n < m_lossDerivativeWithRespectToActFunc[i + 1].size(); n++)
			{
				m_lossDerivativeWithRespectToActFunc[i][j] +=
					m_weights[i + 1][n][j] *
					m_derivatives[i + 1][n] *
					m_lossDerivativeWithRespectToActFunc[i + 1][n];
			}
		}
	}
}

void NeuralNet::updateGradients()
{
	updateWeightsGradients();
	updateBiasesGradients();
}

void NeuralNet::updateWeightsGradients()
{
	for (int i = 0; i < m_weightsGradient.size(); i++)
	{
		for (int n = 0; n < m_weightsGradient[i].size(); n++)
		{
			for (int p = 0; p < m_weightsGradient[i][n].size(); p++)
			{
				if (i == 0)
				{
					m_weightsGradient[i][n][p] +=
						m_val[i][p] *
						m_derivatives[i][n] *
						m_lossDerivativeWithRespectToActFunc[i][n];
				}
				else
				{
					m_weightsGradient[i][n][p] +=
						m_actVal[i - 1][p] *
						m_derivatives[i][n] *
						m_lossDerivativeWithRespectToActFunc[i][n];
				}
			}
		}
	}
}

void NeuralNet::updateBiasesGradients()
{
	for (int i = 0; i < m_biasesGradient.size(); i++)
	{
		for (int j = 0; j < m_biasesGradient[i].size(); j++)
		{
			m_biasesGradient[i][j] +=
				m_derivatives[i][j] *
				m_lossDerivativeWithRespectToActFunc[i][j];
		}
	}
}

void NeuralNet::updateWeights()
{
	for (int i = 0; i < m_weights.size(); i++)
	{
		for (int n = 0; n < m_weights[i].size(); n++)
		{
			for (int p = 0; p < m_weights[i][n].size(); p++)
			{
				m_weights[i][n][p] -= m_learningRate * m_weightsGradient[i][n][p] / m_miniBatchSize;
			}
		}
	}
}

void NeuralNet::updateBiases()
{
	for (int i = 0; i < m_biases.size(); i++)
	{
		for (int j = 0; j < m_biases[i].size(); j++)
		{
			m_biases[i][j] -= m_learningRate * m_biasesGradient[i][j] / m_miniBatchSize;
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
	for (auto& it1 : m_weightsGradient)
	{
		for (auto& it2 : it1)
		{
			for (auto& it3 : it2)
			{
				it3 = 0.0;
			}
		}
	}
}

void NeuralNet::resetBiasesGradients()
{
	for (auto& it1 : m_biasesGradient)
	{
		for (auto& it2 : it1)
		{
			it2 = 0.0;
		}
	}
}

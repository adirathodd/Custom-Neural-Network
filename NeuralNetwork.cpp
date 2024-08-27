#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <numeric>
#include <random>

using namespace std;

inline double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

inline double dSigmoid(double x) {
    return x * (1 - x);
}

NeuralNetwork::NeuralNetwork(int inputs, int hidden, int outputs)
    : numInputs(inputs), numHidden(hidden), numOutputs(outputs) {

    hiddenWeights.resize(numInputs, vector<double>(numHidden));
    outputWeights.resize(numHidden, vector<double>(numOutputs));
    hiddenLayerBias.resize(numHidden);
    outputLayerBias.resize(numOutputs);

    initWeights(hiddenWeights);
    initWeights(outputWeights);
    initBias(hiddenLayerBias);
    initBias(outputLayerBias);
}

void NeuralNetwork::train(const vector<vector<double>>& inputs, const vector<vector<double>>& outputs, int epochs, double lr) {
    vector<int> order(inputs.size());
    iota(order.begin(), order.end(), 0);

    random_device rd;
    mt19937 g(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        shuffle(order.begin(), order.end(), g);

        for (int idx : order) {
            vector<double> hiddenLayer(numHidden);
            vector<double> outputLayer(numOutputs);

            for (int j = 0; j < numHidden; ++j) {
                double activation = hiddenLayerBias[j];
                for (int k = 0; k < numInputs; ++k) {
                    activation += inputs[idx][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            for (int j = 0; j < numOutputs; ++j) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHidden; ++k) {
                    activation += hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }

            vector<double> deltaOutput(numOutputs);
            vector<double> deltaHidden(numHidden);

            for (int j = 0; j < numOutputs; ++j) {
                double error = outputs[idx][j] - outputLayer[j];
                deltaOutput[j] = error * dSigmoid(outputLayer[j]);
            }

            for (int j = 0; j < numHidden; ++j) {
                double error = 0.0;
                for (int k = 0; k < numOutputs; ++k) {
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
            }

            for (int j = 0; j < numOutputs; ++j) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k = 0; k < numHidden; ++k) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }

            for (int j = 0; j < numHidden; ++j) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k = 0; k < numInputs; ++k) {
                    hiddenWeights[k][j] += inputs[idx][k] * deltaHidden[j] * lr;
                }
            }
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> hiddenLayer(numHidden);
    std::vector<double> outputLayer(numOutputs);

    // Compute activations for the hidden layer
    for (int j = 0; j < numHidden; ++j) {
        double activation = hiddenLayerBias[j];
        for (int k = 0; k < numInputs; ++k) {
            activation += input[k] * hiddenWeights[k][j];
        }
        hiddenLayer[j] = sigmoid(activation);
    }

    // Compute activations for the output layer
    for (int j = 0; j < numOutputs; ++j) {
        double activation = outputLayerBias[j];
        for (int k = 0; k < numHidden; ++k) {
            activation += hiddenLayer[k] * outputWeights[k][j];
        }
        outputLayer[j] = sigmoid(activation);
    }

    // Convert the output to a whole integer (0 or 1)
    std::vector<double> binaryOutput(numOutputs);
    for (int i = 0; i < numOutputs; ++i) {
        binaryOutput[i] = (outputLayer[i] >= 0.5) ? 1.0 : 0.0;
    }

    return binaryOutput;
}

void NeuralNetwork::printFinalWeights() const {
    cout << "Final Hidden Weights:\n";
    for (const auto& row : hiddenWeights) {
        for (double weight : row) {
            cout << weight << " ";
        }
        cout << endl;
    }

    cout << "Final Hidden Biases:\n";
    for (double bias : hiddenLayerBias) {
        cout << bias << " ";
    }
    cout << endl;

    cout << "Final Output Weights:\n";
    for (const auto& row : outputWeights) {
        for (double weight : row) {
            cout << weight << " ";
        }
        cout << endl;
    }

    cout << "Final Output Biases:\n";
    for (double bias : outputLayerBias) {
        cout << bias << " ";
    }
    cout << endl;
}

void NeuralNetwork::initWeights(vector<vector<double>>& weights) {
    for (auto& row : weights) {
        for (double& weight : row) {
            weight = static_cast<double>(rand()) / RAND_MAX;
        }
    }
}

void NeuralNetwork::initBias(vector<double>& biases) {
    for (double& bias : biases) {
        bias = static_cast<double>(rand()) / RAND_MAX;
    }
}
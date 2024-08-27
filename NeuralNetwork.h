#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int inputs, int hidden, int outputs);
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& outputs, int epochs, double lr);
    void printFinalWeights() const;
    std::vector<double> predict(const std::vector<double>& input);

private:
    int numInputs, numHidden, numOutputs;

    std::vector<std::vector<double>> hiddenWeights, outputWeights;
    std::vector<double> hiddenLayerBias, outputLayerBias;

    void initWeights(std::vector<std::vector<double>>& weights);
    void initBias(std::vector<double>& biases);
};

#endif
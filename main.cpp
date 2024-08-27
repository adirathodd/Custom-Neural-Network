#include "NeuralNetwork.h"
#include <iostream>

int main() {
    NeuralNetwork nn(2, 2, 1);

    std::vector<std::vector<double>> inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
    };

    std::vector<std::vector<double>> outputs = {
        {1.0},
        {0.0},
        {0.0},
        {1.0}
    };

    nn.train(inputs, outputs, 10000, 0.1);

    nn.printFinalWeights();

    std::cout << "Testing Neural Network:\n";
    for (const auto& input : inputs) {
        std::vector<double> output = nn.predict(input);
        std::cout << "Input: " << input[0] << " " << input[1]
                  << " -> Output: " << output[0] << std::endl;
    }

    return 0;
}
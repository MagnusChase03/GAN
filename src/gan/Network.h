#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.h"
#include <vector>

class Network {
private:
    int inputSize;
    int outputSize;
    int nLayers;

    std::vector<Layer*> layers;
public:
    Network(std::vector<int> shape, ActivationFunction* outputActivation);
    ~Network();

    int getInputSize() {return inputSize;}
    int getOutputSize() {return outputSize;}
    int getNLayers() {return nLayers;}

    Matrix* Forward(Matrix* inputs);
    Matrix* Backward(Matrix* errors, double lr, bool update);

    void Print();
};

#endif

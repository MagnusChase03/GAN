#ifndef LAYER_H
#define LAYER_H

#include "../linalg/Matrix.h"
#include "Activation.h"

class Layer {
private:
    int inputSize;
    int outputSize;
    ActivationFunction* activation;

    Matrix* weights;
    Matrix* bias;

    Matrix* inputCache;
    Matrix* outputCache;
public:
    Layer(int inputSize, int outputSize, ActivationFunction* activation);
    ~Layer();

    int getInputSize() {return inputSize;}
    int getOutputSize() {return outputSize;}
    
    Matrix* Forward(Matrix* inputs);
    Matrix* Backward(Matrix* errors, double lr);
};

#endif

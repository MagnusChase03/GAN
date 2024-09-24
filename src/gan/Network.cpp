#include "Network.h"
#include "Layer.h"
#include <stdexcept>

Network::Network(std::vector<int> shape, ActivationFunction* outputActivation) {
    if (shape.size() < 2) {
        throw std::runtime_error("[ERROR] Invalid shape");
    } 

    this->nLayers = shape.size() - 1;
    this->inputSize = shape[0];
    this->outputSize = shape[this->nLayers];
    this->layers = std::vector<Layer*>(this->nLayers);
    for (int i = 0; i < this->nLayers; i++) {
        if (i == this->nLayers - 1) {
            this->layers[i] = new Layer(shape[i], shape[i + 1], outputActivation);
            break;
        } 
        this->layers[i] = new Layer(shape[i], shape[i + 1], new Sigmoid());
    }
}

Network::~Network() {
    for (int i = 0; i < this->nLayers; i++) {
        delete this->layers[i];
    }
}

Matrix* Network::Forward(Matrix* inputs) {
    if (inputs->getRows() != this->inputSize || inputs->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimensions");
    }

    Matrix* result;
    Matrix* old = nullptr;
    for (int i = 0; i < this->nLayers; i++) {
        if (i == 0) {
            result = this->layers[i]->Forward(inputs);
            continue;
        }
        if (old != nullptr) {
            delete old;
        }
        old = result;
        result = this->layers[i]->Forward(old);
    }
    delete old;
    return result;
}

Matrix* Network::Backward(Matrix* errors, double lr, bool update) {
    if (errors->getRows() != this->outputSize || errors->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimensions");
    }

    Matrix* delta;
    Matrix* old;
    for (int i = this->nLayers - 1; i >= 0; i--) {
        if (i == this->nLayers - 1) {
            delta = this->layers[i]->Backward(errors, lr, update);
            continue;
        }
        if (old != nullptr) {
            delete old;
        }
        old = delta;
        delta = this->layers[i]->Backward(old, lr, update);
    }
    delete old;
    return delta;
}

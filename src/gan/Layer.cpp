#include "Layer.h"
#include "Activation.h"
#include <cstdlib>
#include <ctime>
#include <stdexcept>

Layer::Layer(int inputSize, int outputSize, ActivationFunction* activation) {
    if (inputSize < 1 || outputSize < 1) {
        throw std::runtime_error("[ERROR] Invalid dimensions");
    }

    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->activation = activation;

    std::srand(std::time(nullptr));
    this->weights = new Matrix(this->outputSize, this->inputSize);
    for (int i = 0; i < this->outputSize; i++) {
        for (int j = 0; j < this->inputSize; j++) {
            this->weights->setValue(i, j, (double) std::rand() / RAND_MAX);
        }
    }

    this->bias = new Matrix(this->outputSize, 1);
    this->inputCache = new Matrix(this->inputSize, 1);
    this->outputCache = new Matrix(this->outputSize, 1);
}

Layer::~Layer() {
    delete this->activation;
    delete this->weights;
    delete this->bias;
    delete this->inputCache;
    delete this->outputCache;
}

Matrix* Layer::Forward(Matrix* inputs) {
    if (inputs->getRows() != this->inputSize || inputs->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimensions");
    }

    for (int i = 0; i < inputs->getRows(); i++) {
        this->inputCache->setValue(i, 0, inputs->getValue(i, 0));
    }

    Matrix* product = this->weights->Dot(inputs);
    Matrix* result = product->Add(this->bias); 
    for (int i = 0; i < this->outputSize; i++) {
        this->outputCache->setValue(i, 0, result->getValue(i, 0));
        double post = this->activation->Normal(result->getValue(i, 0));
        result->setValue(i, 0, post);
    }

    return result;
}

Matrix* Layer::Backward(Matrix* errors, double lr, bool update) {
    if (errors->getRows() != this->outputSize || errors->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimensions");
    }

    Matrix* delta = new Matrix(this->inputSize, 1);
    for (int i = 0; i < this->outputSize; i++) {
        for (int j = 0; j < this->inputSize; j++) {
            double prior = this->activation->Dir(this->outputCache->getValue(i, 0));
            double mainDelta = errors->getValue(i, 0) * prior;

            double oldDelta = delta->getValue(j, 0);
            delta->setValue(j, 0, 
                oldDelta + (mainDelta * this->weights->getValue(i, j))
            );

            if (!update) {
                continue;
            }

            double oldWeight = this->weights->getValue(i, j);
            this->weights->setValue(i, j, 
                oldWeight - (mainDelta * this->inputCache->getValue(j, 0) * lr)
            );
        }
    }

    return delta;
}

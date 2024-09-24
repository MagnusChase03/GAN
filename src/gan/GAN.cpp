#include "GAN.h"
#include "Activation.h"
#include <stdexcept>

GAN::GAN(std::vector<int> generatorShape, std::vector<int> discriminatorShape) {
    int generatorEnd = generatorShape[generatorShape.size() - 1];
    int discriminatorStart = discriminatorShape[0];
    int discriminatorEnd = discriminatorShape[discriminatorShape.size() - 1];
    if (generatorEnd != discriminatorStart || discriminatorEnd != 1) {
        throw std::runtime_error("[ERROR] Invalid shape");
    }

    this->generator = new Network(generatorShape, new NoActivation());
    this->discriminator = new Network(discriminatorShape, new Sigmoid());
}

GAN::~GAN() {
    delete this->generator;
    delete this->discriminator;
}

Matrix* GAN::FullForward(Matrix* inputs) {
    if (inputs->getRows() != this->generator->getInputSize() || inputs->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimension");
    } 

    Matrix* generatedOuput = this->generator->Forward(inputs);
    Matrix* classificationOutput = this->discriminator->Forward(generatedOuput);
    delete generatedOuput;
    return classificationOutput;
}

Matrix* GAN::GeneratorForward(Matrix* inputs) {
    if (inputs->getRows() != this->generator->getInputSize() || inputs->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimension");
    } 

    Matrix* generatedOuput = this->generator->Forward(inputs);
    return generatedOuput;
}

Matrix* GAN::DiscriminatorForward(Matrix* inputs) {
    if (inputs->getRows() != this->discriminator->getInputSize() || inputs->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimension");
    } 

    Matrix* classificationOutput = this->discriminator->Forward(inputs);
    return classificationOutput;
}

Matrix* GAN::GeneratorBackward(Matrix* errors, double lr, bool update) {
    if (errors->getRows() != this->generator->getOutputSize() && errors->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimension");
    }

    Matrix* delta = this->generator->Backward(errors, lr, update);
    return delta;
}

Matrix* GAN::DiscriminatorBackward(Matrix* errors, double lr, bool update) {
    if (errors->getRows() != 1 && errors->getCols() != 1) {
        throw std::runtime_error("[ERROR] Invalid dimension");
    }

    Matrix* delta = this->discriminator->Backward(errors, lr, update);
    return delta;
}

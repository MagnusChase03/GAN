#include "GAN.h"
#include "Activation.h"
#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>

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

Matrix* GAN::Generate(int num) {
    int generatorInputSize = this->generator->getInputSize();
    Matrix* result = new Matrix(num, this->generator->getOutputSize());
    for (int i = 0; i < num; i++) {
        std::vector<std::vector<double>> fakeInput = std::vector<std::vector<double>>(generatorInputSize);
        for (int k = 0; k < generatorInputSize; k++) {
            fakeInput[k] = {(double) std::rand() / RAND_MAX};
        }

        Matrix* fakeInputMatrix = new Matrix(fakeInput);
        Matrix* fakeGenerated = this->GeneratorForward(fakeInputMatrix);
        for (int j = 0; j < this->generator->getOutputSize(); j++) {
            result->setValue(i, j, fakeGenerated->getValue(j, 0));
        }

        delete fakeGenerated;
        delete fakeInputMatrix;

    }
    return result;
}

void GAN::Train(Matrix* realData, int iterations, double lr) {
    std::srand(std::time(nullptr));
    int generatorInputSize = this->generator->getInputSize();
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < realData->getRows(); j++) {
            Matrix* realInput = realData->TakeRow(j);
            Matrix* realOutput = this->DiscriminatorForward(realInput);
            double realClass = realOutput->getValue(0, 0);

            printf("Real Data Error: %.4f ", -std::log(realClass));
            std::vector<std::vector<double>> realError = {{-1.0 / realClass}};
            Matrix* realErrorMatrix = new Matrix(realError);
            delete this->DiscriminatorBackward(realErrorMatrix, lr, true);

            std::vector<std::vector<double>> fakeInput = std::vector<std::vector<double>>(generatorInputSize);
            for (int k = 0; k < generatorInputSize; k++) {
                fakeInput[k] = {(double) std::rand() / RAND_MAX};
            }
            Matrix* fakeInputMatrix = new Matrix(fakeInput);
            Matrix* fakeGenerated = this->GeneratorForward(fakeInputMatrix);
            Matrix* fakeOutput = this->DiscriminatorForward(fakeGenerated);
            double fakeClass = fakeOutput->getValue(0, 0);
            printf("Fake Data Error: %.4f\n", -std::log(fakeClass));
            if (fakeClass >= 0.5) {
                std::vector<std::vector<double>> fakeError = {{1.0 / (1.0 - fakeClass)}};
                Matrix* fakeErrorMatrix = new Matrix(fakeError);
                delete this->DiscriminatorBackward(fakeErrorMatrix, lr, true);
                delete fakeErrorMatrix;
            } else {
                std::vector<std::vector<double>> fakeError = {{-1.0 / fakeClass}};
                Matrix* fakeErrorMatrix = new Matrix(fakeError);
                Matrix* delta = this->DiscriminatorBackward(fakeErrorMatrix, lr, false);
                delete this->GeneratorBackward(delta, lr, true);
                delete delta;
                delete fakeErrorMatrix;
            }

            fakeGenerated->Print();
            delete fakeOutput;
            delete fakeGenerated;
            delete fakeInputMatrix;
            delete realErrorMatrix;
            delete realOutput;
            delete realInput;
        }
    }
}

void GAN::Print() {
    printf("Generator\n---\n");
    this->generator->Print();
    printf("Discriminator\n---\n");
    this->discriminator->Print();
}

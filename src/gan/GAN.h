#ifndef GAN_H
#define GAN_H

#include "Network.h"

class GAN {
private:
    Network* generator;
    Network* discriminator;
public:
    GAN(std::vector<int> generatorShape, std::vector<int> discriminatorShape);
    ~GAN();

    Matrix* FullForward(Matrix* inputs);
    Matrix* GeneratorForward(Matrix* inputs);
    Matrix* DiscriminatorForward(Matrix* inputs);

    Matrix* GeneratorBackward(Matrix* errors, double lr, bool update);
    Matrix* DiscriminatorBackward(Matrix* errors, double lr, bool update);

    void Print();
};

#endif

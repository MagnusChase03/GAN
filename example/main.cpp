#include <gan/GAN.h>
#include <linalg/Matrix.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(std::time(nullptr));
    std::vector<int> generatorShape = {1, 2, 2, 4};
    std::vector<int> discriminatorShape = {4, 2, 2, 1};
    GAN* gan = new GAN(generatorShape, discriminatorShape);

    std::vector<std::vector<double>> realValues = {{1.0}, {0.0}, {0.0}, {1.0}};
    Matrix* realValuesMatrix = new Matrix(realValues);
    std::vector<std::vector<double>> realValues2 = {{0.0}, {1.0}, {1.0}, {0.0}};
    Matrix* realValuesMatrix2 = new Matrix(realValues2);
    for (int i = 0; i < 10; i++) {
        printf("-----\n");
        Matrix* output = gan->DiscriminatorForward(realValuesMatrix);
        output->Print();

        std::vector<std::vector<double>> error = {{output->getValue(0, 0) - 1.0}};
        Matrix* errorM = new Matrix(error);
        gan->DiscriminatorBackward(errorM, 0.1, true);

        std::vector<std::vector<double>> inputs = {{(double) std::rand() / RAND_MAX}};
        Matrix* inputX = new Matrix(inputs);
        Matrix* generated = gan->GeneratorForward(inputX);
        printf("===\n");
        generated->Print();
        printf("===\n");
        output = gan->DiscriminatorForward(generated);
        output->Print();
        double classification = output->getValue(0, 0);
        if (classification >= 0.5) {
            error = {{classification}};
            errorM = new Matrix(error);
            gan->DiscriminatorBackward(errorM, 0.1, true);
        } else {
            error = {{classification - 1.0}};
            errorM = new Matrix(error);
            Matrix* delta = gan->DiscriminatorBackward(errorM, 1.0, false);
            gan->GeneratorBackward(delta, 0.1, true);
        }

        output = gan->DiscriminatorForward(realValuesMatrix2);
        output->Print();

        error = {{output->getValue(0, 0)}};
        errorM = new Matrix(error);
        gan->DiscriminatorBackward(errorM, 0.1, true);
        printf("-----\n");
    }

    return 0;
}

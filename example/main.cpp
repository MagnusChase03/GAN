#include <gan/GAN.h>
#include <linalg/Matrix.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

int main() {
    std::srand(std::time(nullptr));
    std::vector<int> generatorShape = {1, 4};
    std::vector<int> discriminatorShape = {4, 1};
    GAN* gan = new GAN(generatorShape, discriminatorShape);

    std::vector<std::vector<double>> realValues = {{1.0}, {0.1}, {0.1}, {1.0}};
    Matrix* realValuesMatrix = new Matrix(realValues);
    for (int i = 0; i < 1000; i++) {
        printf("-----\n");
        Matrix* output = gan->DiscriminatorForward(realValuesMatrix);

        printf("Real Error: %.4f\n", -std::log(output->getValue(0, 0)));
        std::vector<std::vector<double>> error = {{-1.0 / output->getValue(0, 0)}};
        Matrix* errorM = new Matrix(error);
        gan->DiscriminatorBackward(errorM, 0.01, true);

        std::vector<std::vector<double>> inputs = {{(double) std::rand() / RAND_MAX}};
        Matrix* inputX = new Matrix(inputs);
        Matrix* generated = gan->GeneratorForward(inputX);
        generated->Print();
        output = gan->DiscriminatorForward(generated);
        double classification = output->getValue(0, 0);
        printf("Fake Error: %.4f\n", -std::log(1.0 - classification));
        if (classification < 0.5) {
            printf("Updating bad generator\n");
            error = {{-1.0 / classification}};
            errorM = new Matrix(error);
            Matrix* delta = gan->DiscriminatorBackward(errorM, 1.0, false);
            gan->GeneratorBackward(delta, 0.01, true);
        } else {
            printf("Updating bad discriminator\n");
            error = {{1.0 / (1.0 - classification)}};
            errorM = new Matrix(error);
            gan->DiscriminatorBackward(errorM, 0.01, true);

        }
        printf("-----\n");
    }
    //gan->Print();

    return 0;
}

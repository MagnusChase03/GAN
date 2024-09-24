#include <gan/GAN.h>
#include <linalg/Matrix.h>
#include <vector>

int main() {
    std::vector<int> generatorShape = {1, 2, 4};
    std::vector<int> discriminatorShape = {4, 2, 1};
    GAN* gan = new GAN(generatorShape, discriminatorShape);

    std::vector<std::vector<double>> inputs = {{0.61}};
    Matrix* inputMatrix = new Matrix(inputs);
    Matrix* outputs = gan->GeneratorForward(inputMatrix);
    outputs->Print();

    return 0;
}

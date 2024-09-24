#include <gan/GAN.h>
#include <linalg/Matrix.h>
#include <vector>

int main() {
    std::vector<int> generatorShape = {1, 2, 4};
    std::vector<int> discriminatorShape = {4, 2, 1};
    GAN* gan = new GAN(generatorShape, discriminatorShape);

    std::vector<std::vector<double>> realValues = {{1.0}, {0.0}, {0.0}, {1.0}};
    Matrix* realValuesMatrix = new Matrix(realValues);
    for (int i = 0; i < 10; i++) {
        Matrix* output = gan->DiscriminatorForward(realValuesMatrix);
        output->Print();

        std::vector<std::vector<double>> error = {{output->getValue(0, 0) - 1.0}};
        Matrix* errorM = new Matrix(error);
        gan->DiscriminatorBackward(errorM, 1.0, true);
        
        delete errorM;
        delete output;
    }

    return 0;
}

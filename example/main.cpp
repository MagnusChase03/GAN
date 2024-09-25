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

    std::vector<std::vector<double>> realValues = {{1.0, 0.1, 0.1, 1.0}};
    Matrix* realValuesMatrix = new Matrix(realValues);
    gan->Train(realValuesMatrix, 1000, 0.01);
    //gan->Print();

    return 0;
}

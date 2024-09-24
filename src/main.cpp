#include "gan/Network.h"
#include "gan/Activation.h"
#include "linalg/Matrix.h"
#include <vector>

int main() {
    std::vector<std::vector<double>> inputs = {{1.0}, {2.0}, {3.0}};
    Matrix* inputMatrix = new Matrix(inputs);
    std::vector<std::vector<double>> errors = {{-1.0}, {1.0}};
    Matrix* errorMatrix = new Matrix(errors);

    std::vector<int> shape = {3, 2, 2};
    Network* n = new Network(shape, new NoActivation());
    Matrix* output = n->Forward(inputMatrix);
    output->print();
    n->Backward(errorMatrix, 1.0);
    Matrix* output2 = n->Forward(inputMatrix);
    output2->print();
}

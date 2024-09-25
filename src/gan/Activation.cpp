#include "Activation.h"
#include <cmath>

double Sigmoid::Normal(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::Dir(double x) {
    return this->Normal(x) * (1.0 - this->Normal(x));
}

double Relu::Normal(double x) {
    if (x >= 0) {
        return x;
    }
    return 0.1 * x;
}

double Relu::Dir(double x) {
    if (x >= 0) {
        return 1;
    }
    return 0.1;
}

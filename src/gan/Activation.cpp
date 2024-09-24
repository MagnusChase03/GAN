#include "Activation.h"
#include <cmath>

double Sigmoid::normal(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Sigmoid::dir(double x) {
    return this->normal(x) * (1.0 - this->normal(x));
}

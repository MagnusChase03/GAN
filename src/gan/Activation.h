#ifndef ACTIVATION_H
#define ACTIVATION_H

class ActivationFunction {
public:
    virtual ~ActivationFunction() {};
    virtual double normal(double x) = 0;
    virtual double dir(double x) = 0;
};

class Sigmoid : public ActivationFunction {
public:
    ~Sigmoid() {}
    double normal(double x);
    double dir(double x);
};

class NoActivation : public ActivationFunction {
public:
    ~NoActivation() {}
    double normal(double x) {return x;}
    double dir(double x) {return 1.0;}
};

#endif

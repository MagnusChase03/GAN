#ifndef ACTIVATION_H
#define ACTIVATION_H

class ActivationFunction {
public:
    virtual ~ActivationFunction() {};
    virtual double Normal(double x) = 0;
    virtual double Dir(double x) = 0;
};

class Sigmoid : public ActivationFunction {
public:
    ~Sigmoid() {}
    double Normal(double x);
    double Dir(double x);
};

class NoActivation : public ActivationFunction {
public:
    ~NoActivation() {}
    double Normal(double x) {return x;}
    double Dir(double x) {return 1.0;}
};

#endif

#ifndef SIGMOID_H
#define SIGMOID_H

#include "math/ActivationFunction.hpp"
#include <cmath>
class Sigmoid : ActivationFunction {
public:
  float activate(float input) const override {
    return 1.0f / (1 + std::exp(-input));
  }
  float deriv(float input) const override {
    //(1 + e^-x)^-1
    //-(1+e^(-x))^(-2) * e^(-x) * (-1)
    return std::pow(1 + std::exp(-input), -2) * std::exp(-input);
  }
};

#endif // !SIGMOID_H

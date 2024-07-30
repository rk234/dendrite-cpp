#ifndef SIGMOID_H
#define SIGMOID_H

#include "math/ActivationFunction.hpp"
#include <cmath>
class Sigmoid : ActivationFunction {
private:
  static float sigmoid(float x) { return 1.0f / (1 + std::expf(-x)); }
  static float sigmoid_deriv(float x) {
    //(1 + e^-x)^-1
    //-(1+e^(-x))^(-2) * e^(-x) * (-1)
    return std::pow(1 + std::expf(-x), -2) * std::expf(-x);
  }

public:
  Matrix activate(const Matrix &input) const override {
    return input.apply_function(sigmoid);
  }

  Matrix deriv(const Matrix &input) const override {
    return input.apply_function(sigmoid_deriv);
  }

  Matrix &activate_inplace(Matrix &input) const override {
    input.apply_function_inplace(sigmoid);
    return input;
  }

  Matrix &deriv_inplace(Matrix &input) const override {
    input.apply_function_inplace(sigmoid_deriv);
    return input;
  }
};

#endif // !SIGMOID_H

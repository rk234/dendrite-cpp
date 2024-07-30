#ifndef ReLU_H
#define ReLU_H

#include "Matrix.hpp"
#include "math/ActivationFunction.hpp"
class ReLU : ActivationFunction {
private:
  static float relu(float x) {
    if (x >= 0) {
      return x;
    } else {
      return 0;
    }
  }

  static float relu_deriv(float x) {
    if (x >= 0) {
      return 1;
    } else {
      return 0;
    }
  }

public:
  Matrix activate(const Matrix &input) const override {
    return input.apply_function(relu);
  }

  Matrix deriv(const Matrix &input) const override {
    return input.apply_function(relu_deriv);
  }

  Matrix &activate_inplace(Matrix &input) const override {
    input.apply_function_inplace(relu);
    return input;
  }

  Matrix &deriv_inplace(Matrix &input) const override {
    input.apply_function_inplace(relu_deriv);
    return input;
  }
};

#endif // !ReLU_H

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "math/ActivationFunction.hpp"
#include <cmath>
class Sigmoid : ActivationFunction {
public:
  Matrix activate(const Matrix &input) const override {
    // TODO: implement softmax and deriv
  }

  Matrix deriv(const Matrix &input) const override {
    // TODO: implement softmax and deriv
  }

  Matrix &activate_inplace(Matrix &input) const override {
    // TODO: implement softmax and deriv
  }

  Matrix &deriv_inplace(Matrix &input) const override {
    // TODO: implement softmax and deriv
  }
};

#endif // !SOFTMAX_H

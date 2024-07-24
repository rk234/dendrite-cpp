#ifndef ReLU_H
#define ReLU_H

#include "math/ActivationFunction.hpp"
class ReLU : ActivationFunction {
public:
  float activate(float input) override {
    if (input < 0) {
      return input;
    } else {
      return 0;
    }
  }

  float deriv(float input) override {
    if (input < 0) {
      return 0;
    } else {
      return 1;
    }
  }
};

#endif // !ReLU_H

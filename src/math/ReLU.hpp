#ifndef ReLU_H
#define ReLU_H

#include "nn/ActivationFunction.hpp"
class ReLU : ActivationFunction {
public:
  float activate(float input) override {
    if (input < 0) {
      return input;
    } else {
      return 0;
    }
  }
};

#endif // !ReLU_H

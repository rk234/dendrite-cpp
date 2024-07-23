#ifndef SIGMOID_H
#define SIGMOID_H

#include "nn/ActivationFunction.hpp"
#include <cmath>
class Sigmoid : ActivationFunction {
public:
  float activate(float input) override { return 1.0f / (1 + std::exp(-input)); }
};

#endif // !SIGMOID_H

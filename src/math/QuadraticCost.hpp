#ifndef QE_H
#define QE_H

#include "math/CostFunction.hpp"
#include <cmath>
class QuadraticCost : public CostFunction {
public:
  Matrix cost(Matrix &x, Matrix &truth) const override {
    return 0.5f * ((x - truth).pow_elem_inplace(2));
  }

  Matrix deriv(Matrix &x, Matrix &truth) const override { return (x - truth); }
};

#endif // !QE_H

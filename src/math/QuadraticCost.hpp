#ifndef QE_H
#define QE_H

#include "math/CostFunction.hpp"
#include <cmath>
class QuadraticCost : public CostFunction {
public:
  Matrix cost(Matrix &x, Matrix &truth) const override {
    return ((x - truth).pow_elem_inplace(2)).scale(0.5f);
  }

  Matrix deriv(Matrix &x, Matrix &truth) const override { return (x - truth); }
};

#endif // !QE_H

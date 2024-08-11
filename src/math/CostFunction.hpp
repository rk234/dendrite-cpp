#ifndef COST_H
#define COST_H

#include "math/Matrix.hpp"
class CostFunction {
public:
  virtual Matrix cost(const Matrix &x, const Matrix &truth) const = 0;
  virtual Matrix deriv(const Matrix &x, const Matrix &truth) const = 0;
};

#endif // !COST_H

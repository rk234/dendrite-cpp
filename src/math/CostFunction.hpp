#ifndef COST_H
#define COST_H

#include "math/Matrix.hpp"
class CostFunction {
public:
  virtual Matrix cost(Matrix &x, Matrix &truth) const = 0;
  virtual Matrix deriv(Matrix &x, Matrix &truth) const = 0;
};

#endif // !COST_H

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "math/Matrix.hpp"

class ActivationFunction {
public:
  virtual Matrix activate(const Matrix &input) const = 0;
  virtual Matrix &activate_inplace(Matrix &input) const = 0;
  virtual Matrix deriv(const Matrix &input) const = 0;
  virtual Matrix &deriv_inplace(Matrix &input) const = 0;
};
#endif // !ACTIVATION_H

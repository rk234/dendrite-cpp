#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "math/Matrix.hpp"
#include <map>

class ActivationFunction {
public:
  virtual Matrix activate(const Matrix &input) const = 0;
  virtual Matrix &activate_inplace(Matrix &input) const = 0;
  virtual Matrix deriv(const Matrix &input) const = 0;
  virtual Matrix &deriv_inplace(Matrix &input) const = 0;

  inline static std::map<std::string, ActivationFunction *>
      s_activationFunctions;

  static void register_func(const std::string &name,
                            ActivationFunction *activationFn) {
    s_activationFunctions[name] = activationFn;
  }

  static ActivationFunction &get_from_name(const std::string &name) {
    return *s_activationFunctions[name];
  }
};

#endif // !ACTIVATION_H

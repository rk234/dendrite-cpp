#ifndef COST_H
#define COST_H

#include "math/Matrix.hpp"
#include <map>
#include <string>

namespace Dendrite {
class CostFunction {
public:
  virtual Matrix cost(const Matrix &x, const Matrix &truth) const = 0;
  virtual Matrix deriv(const Matrix &x, const Matrix &truth) const = 0;

  inline static std::map<std::string, CostFunction *> s_costFunctions;

  static CostFunction &get_from_name(const std::string &name) {
    return *s_costFunctions[name];
  }

  static void register_func(const std::string &name, CostFunction *cost) {
    s_costFunctions[name] = cost;
  }
};
} // namespace Dendrite

#endif // !COST_H

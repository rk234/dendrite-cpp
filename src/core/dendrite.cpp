#include "dendrite.hpp"
#include "math/ActivationFunction.hpp"
#include "math/CostFunction.hpp"
#include "math/QuadraticCost.hpp"
#include "math/ReLU.hpp"
#include "math/Sigmoid.hpp"
#include "math/Softmax.hpp"

namespace Dendrite {
void init_functions() {
  CostFunction::register_func("quadratic",
                              (CostFunction *)(new QuadraticCost()));

  ActivationFunction::register_func("sigmoid",
                                    (ActivationFunction *)(new Sigmoid()));
  ActivationFunction::register_func("relu", (ActivationFunction *)(new ReLU()));
  ActivationFunction::register_func("softmax",
                                    (ActivationFunction *)(new Softmax()));
}
} // namespace Dendrite

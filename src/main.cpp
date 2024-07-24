#include "math/ActivationFunction.hpp"
#include "math/Matrix.hpp"
#include "nn/NeuralNetwork.hpp"

int main() {
  // clang-format off
  float m1[1][3] = {
    {1, 2, 3}
  };
  float m2[3][2] = {
    {10, 11},
    {20, 21},
    {30, 31}
  };
  // clang-format on

  Matrix mat = Matrix(m1);

  NeuralNetwork net = NeuralNetwork();
  ActivationFunction *relu = (ActivationFunction *)(new ReLU());
  net.set_input_layer(4);
  net.add_hidden_layer(6, *relu);
  net.set_output_layer(3, *relu);

  net.forward(mat).print();
}

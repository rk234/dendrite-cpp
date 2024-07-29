#include "math/ActivationFunction.hpp"
#include "math/Matrix.hpp"
#include "math/ReLU.hpp"
#include "nn/NeuralNetwork.hpp"
#include "testing/Mnist.hpp"
#include <filesystem>

int main() {
  // clang-format off
  float m1[4][1] = {
    {1},
    {2},
    {3},
    {4}
  };
  float m2[3][2] = {
    {10, 11},
    {20, 21},
    {30, 31}
  };
  // clang-format on

  Matrix mat = Matrix(m1);

  // NeuralNetwork net = NeuralNetwork();
  // ActivationFunction *relu = (ActivationFunction *)(new ReLU());
  // net.set_input_layer(4);
  // net.add_hidden_layer(6, *relu);
  // net.add_hidden_layer(5, *relu);
  // net.add_hidden_layer(6, *relu);
  // net.set_output_layer(4, *relu);
  // net.init();
  //
  // net.forward(mat).print();

  Mnist mnist = Mnist();
  mnist.load(std::filesystem::path("res/MNIST"));
}

#include "math/ActivationFunction.hpp"
#include "math/CostFunction.hpp"
#include "math/Matrix.hpp"
#include "math/QuadraticCost.hpp"
#include "math/ReLU.hpp"
#include "math/Sigmoid.hpp"
#include "nn/NeuralNetwork.hpp"
#include "testing/Mnist.hpp"
#include <filesystem>

int main() {
  Mnist mnist = Mnist();
  mnist.load(std::filesystem::path("res/MNIST"));

  const Matrix trainImages = mnist.get_train_images().value();
  const Matrix trainLabels = mnist.get_train_labels().value();

  CostFunction *costFn = (CostFunction *)(new QuadraticCost());

  NeuralNetwork net = NeuralNetwork(*costFn);
  ActivationFunction *relu = (ActivationFunction *)(new ReLU());
  ActivationFunction *sigmoid = (ActivationFunction *)(new Sigmoid());
  net.set_input_layer(trainImages.rows());
  net.add_hidden_layer(128, *sigmoid);
  net.add_hidden_layer(64, *sigmoid);
  net.set_output_layer(10, *sigmoid);
  net.init();

  net.update_batch(trainImages, trainLabels, 0.05);

  // net.forward(trainImages.get_col(0)).print();
}

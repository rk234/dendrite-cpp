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
  NeuralNetwork::init_functions();
  Mnist mnist = Mnist();
  mnist.load(std::filesystem::path("res/MNIST"));

  const Matrix trainImages = mnist.get_train_images().value();
  const Matrix trainLabels = mnist.get_train_labels().value();

  NeuralNetwork net = NeuralNetwork(CostFunction::get_from_name("quadratic"));
  net.set_input_layer(trainImages.rows());
  net.add_hidden_layer(128, ActivationFunction::get_from_name("sigmoid"));
  net.add_hidden_layer(64, ActivationFunction::get_from_name("sigmoid"));
  net.set_output_layer(10, ActivationFunction::get_from_name("sigmoid"));
  net.init();

  // 120 epochs ~~> 0.80-0.90 accuracy!
  net.train(trainImages, trainLabels, 100);

  // net.forward(trainImages.get_col(0)).print();
}

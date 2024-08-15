#include "core/dendrite.hpp"
#include "math/Matrix.hpp"
#include "nn/NeuralNetwork.hpp"
#include "testing/Mnist.hpp"
#include <filesystem>

int main() {
  Dendrite::init_functions();

  Mnist mnist = Mnist();
  mnist.load(std::filesystem::path("res/MNIST"));

  const Matrix trainImages = mnist.get_train_images().value();
  const Matrix trainLabels = mnist.get_train_labels().value();

  NeuralNetwork net = NeuralNetwork("quadratic");
  net.set_input_layer(trainImages.rows());
  net.add_hidden_layer(128, ("sigmoid"));
  net.add_hidden_layer(64, ("sigmoid"));
  net.set_output_layer(10, ("sigmoid"));
  net.init();

  net.forward(trainImages.get_col(0)).print();
  net.save("res/models/test.dm");

  NeuralNetwork loadedNet = NeuralNetwork();
  loadedNet.load("res/models/test.dm");
  loadedNet.forward(trainImages.get_col(0)).print();

  // 120 epochs ~~> 0.80-0.90 accuracy!
  // net.train(trainImages, trainLabels, 100);
}

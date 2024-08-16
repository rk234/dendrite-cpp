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

  NeuralNetwork net = NeuralNetwork();
  // net.set_input_layer(trainImages.rows());
  // net.add_hidden_layer(128, ("sigmoid"));
  // net.add_hidden_layer(64, ("sigmoid"));
  // net.set_output_layer(10, ("sigmoid"));
  // net.init();
  //
  // net.train(trainImages, trainLabels, 100, 140, 0.05);
  // net.save("res/models/test.dm");
  //
  //
  net.load("res/models/test.dm");
  const Matrix testImages = mnist.get_test_images().value();
  const Matrix testLabels = mnist.get_test_labels().value();

  mnist.display_image(testImages, 28, 0);

  std::cout << "MODEL PREDICTION: \n";

  net.forward(testImages.get_col(0)).print();
}

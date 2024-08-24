#include "core/dendrite.hpp"
#include "math/Matrix.hpp"
#include "nn/NeuralNetwork.hpp"
#include "testing/Mnist.hpp"
#include <filesystem>

bool check_one_hot(const Dendrite::Matrix &pred,
                   const Dendrite::Matrix &truth) {
  size_t predIdx = -1;
  float predMax = 0;
  size_t truthIdx = -1;

  for (size_t i = 0; i < truth.rows(); i++)
    if (truth.get(i, 0) == 1.0f)
      truthIdx = i;

  for (size_t i = 0; i < pred.rows(); i++) {
    float x = pred.get(i, 0);
    if (x > predMax) {
      predMax = x;
      predIdx = i;
    }
  }

  return predIdx == truthIdx;
}

int main() {
  Dendrite::init_functions();

  Mnist mnist = Mnist();
  mnist.load(std::filesystem::path("res/MNIST"));

  const Dendrite::Matrix trainImages = mnist.get_train_images().value();
  const Dendrite::Matrix trainLabels = mnist.get_train_labels().value();

  Dendrite::NeuralNetwork net = Dendrite::NeuralNetwork("quadratic");
  net.set_input_layer(trainImages.rows());
  net.add_hidden_layer(128, ("sigmoid"));
  net.add_hidden_layer(64, ("sigmoid"));
  net.set_output_layer(10, ("sigmoid"));
  net.init();

  net.train(trainImages, trainLabels, 100, 200, 0.05);
  net.save("res/models/test.dm");

  net.load("res/models/test.dm");
  const Dendrite::Matrix testImages = mnist.get_test_images().value();
  const Dendrite::Matrix testLabels = mnist.get_test_labels().value();

  size_t correct = 0;

  for (size_t i = 0; i < testImages.cols(); i++) {
    std::cout << "correct\n";
    Dendrite::Matrix out = net.forward(testImages.get_col(i));
    Dendrite::Matrix truth = testLabels.get_col(i);

    if (check_one_hot(out, truth)) {
      correct++;
    }
  }

  std::cout << "ACCURACY ON TEST DATA: " << ((float)correct / testLabels.cols())
            << "\n";
}

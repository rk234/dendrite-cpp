#ifndef NN_H
#define NN_H

#include "math/ActivationFunction.hpp"
#include "math/CostFunction.hpp"
#include "math/Matrix.hpp"
#include "nn/Layer.hpp"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <memory>
#include <random>
#include <string>

namespace Dendrite {
class NeuralNetwork {
private:
  std::vector<std::shared_ptr<HiddenLayer>> m_hiddenLayers;
  std::shared_ptr<OutputLayer> m_outputLayer;
  std::shared_ptr<InputLayer> m_inputLayer;
  std::string m_costFunction;

  std::tuple<Matrix, Matrix> shuffle_train(const Matrix &trainX,
                                           const Matrix &trainY) {
    assert(trainX.cols() == trainY.cols());

    std::vector<size_t> indices = std::vector<size_t>(trainX.cols());
    for (size_t i = 0; i < trainX.cols(); i++) {
      indices.push_back(i);
    }
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});

    Matrix trainXS = Matrix::with_same_shape(trainX);
    Matrix trainYS = Matrix::with_same_shape(trainY);

    for (size_t i = 0; i < trainX.cols(); i++) {
      for (size_t r = 0; r < trainX.rows(); r++) {
        trainXS.set(r, i, trainX.get(r, indices[i]));
      }
      for (size_t r = 0; r < trainY.rows(); r++) {
        trainYS.set(r, i, trainY.get(r, indices[i]));
      }
    }

    return std::tuple<Matrix, Matrix>(trainXS, trainYS);
  }

public:
  NeuralNetwork(const NeuralNetwork &other)
      : m_costFunction(other.m_costFunction) {}
  NeuralNetwork(const std::string costFn) : m_costFunction(costFn) {}
  NeuralNetwork() {}

  void set_input_layer(int numInputs);

  void add_hidden_layer(int numNeurons, const std::string fn);

  void set_output_layer(int numOutputs, const std::string fn);

  void init();

  Matrix forward(const Matrix &inputs);

  void update_batch(const Matrix &xs, const Matrix &ys, size_t start,
                    size_t end, float learningRate);

  std::tuple<std::vector<Matrix>, std::vector<Matrix>>
  backprop(const Matrix &xs, const Matrix &ys, size_t exampleIndex);

  void train(const Matrix &trainX, const Matrix &trainY, size_t batchSize,
             size_t epochs, float learningRate);

  void save(std::filesystem::path outPath);

  void load(std::filesystem::path path);

  size_t num_layers() const;
};
} // namespace Dendrite

#endif

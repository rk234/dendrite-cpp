#ifndef NN_H
#define NN_H

#include "math/ActivationFunction.hpp"
#include "math/CostFunction.hpp"
#include "math/Matrix.hpp"
#include "nn/Layer.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

class NeuralNetwork {
private:
  std::vector<std::shared_ptr<HiddenLayer>> m_hiddenLayers;
  std::shared_ptr<OutputLayer> m_outputLayer;
  std::shared_ptr<InputLayer> m_inputLayer;
  std::string m_costFunction;

public:
  NeuralNetwork(const NeuralNetwork &other)
      : m_costFunction(other.m_costFunction) {}
  NeuralNetwork(const std::string costFn) : m_costFunction(costFn) {}
  NeuralNetwork() {}

  void set_input_layer(int numInputs) {
    this->m_inputLayer = std::make_shared<InputLayer>(numInputs);
  }

  void add_hidden_layer(int numNeurons, const std::string fn) {
    assert(m_inputLayer);
    std::shared_ptr<Layer> prev;
    if (m_hiddenLayers.size() > 0) {
      prev = m_hiddenLayers[m_hiddenLayers.size() - 1];
    } else {
      prev = std::shared_ptr<Layer>(m_inputLayer);
    }

    m_hiddenLayers.push_back(
        std::make_shared<HiddenLayer>(numNeurons, prev, fn));
  }

  void set_output_layer(int numOutputs, const std::string fn) {
    assert(m_inputLayer);
    std::shared_ptr<Layer> prev;
    if (m_hiddenLayers.size() > 0) {
      prev = m_hiddenLayers[m_hiddenLayers.size() - 1];
    } else {
      prev = std::shared_ptr<Layer>(m_inputLayer);
    }

    this->m_outputLayer = std::make_shared<OutputLayer>(numOutputs, prev, fn);
  }

  void init() {
    for (size_t i = 0; i < m_hiddenLayers.size(); i++) {
      m_hiddenLayers[i]->rand_init();
    }
    m_outputLayer->rand_init();
  }

  Matrix forward(const Matrix &inputs) {
    m_inputLayer->set_inputs(inputs);
    for (size_t i = 0; i < m_hiddenLayers.size(); i++) {
      // std::cout << "=====Calculating Activations For Layer " << i << "\n";
      m_hiddenLayers[i]->calc_activations();
    }
    return m_outputLayer->calc_outputs();
  }

  void update_batch(const Matrix &xs, const Matrix &ys, size_t start,
                    size_t end,
                    float learningRate) { // Columns in x and y should be
                                          // inputs/output vectors
    assert(xs.cols() == ys.cols());
    assert(m_inputLayer);
    assert(m_outputLayer);

    std::vector<Matrix> layerWeightGradients;
    std::vector<Matrix> layerBiasGradients;

    for (size_t i = 0; i < m_hiddenLayers.size(); i++) {
      layerBiasGradients.emplace_back(
          Matrix::with_same_shape(m_hiddenLayers[i]->m_bias));
      layerWeightGradients.emplace_back(
          Matrix::with_same_shape(m_hiddenLayers[i]->m_weights));
    }
    layerBiasGradients.emplace_back(
        Matrix::with_same_shape(m_outputLayer->m_bias));
    layerWeightGradients.emplace_back(
        Matrix::with_same_shape(m_outputLayer->m_weights));

    // calculate the gradients for each weight bias matrix per layer
    for (size_t i = start; i < end; i++) {
      const auto [dw, db] = backprop(xs, ys, i); // weights, biases

      for (size_t j = 0; j < layerWeightGradients.size(); j++) {
        layerWeightGradients[j] += dw[j];
        layerBiasGradients[j] += db[j];
      }
    }

    size_t n = end - start;

    // apply the gradients to each weight/bias matrix per layer
    for (size_t i = 0; i < m_hiddenLayers.size(); i++) {
      m_hiddenLayers[i]->m_weights -=
          (layerWeightGradients[i] * (learningRate / n));
      m_hiddenLayers[i]->m_bias -= (layerBiasGradients[i] * (learningRate / n));
    }
    m_outputLayer->m_weights -=
        (layerWeightGradients[layerWeightGradients.size() - 1] *
         (learningRate / n));
    m_outputLayer->m_bias -=
        (layerBiasGradients[layerWeightGradients.size() - 1] *
         (learningRate / n));
  }

  std::tuple<std::vector<Matrix>, std::vector<Matrix>>
  backprop(const Matrix &xs, const Matrix &ys,
           size_t exampleIndex) { // Columns in x and y should be inputs/output
                                  // vectors
    assert(xs.cols() == ys.cols());
    assert(exampleIndex >= 0 && exampleIndex < xs.cols());

    std::vector<Matrix> weightGradients;
    std::vector<Matrix> biasGradients;

    for (size_t i = 0; i < m_hiddenLayers.size(); i++) {
      biasGradients.emplace_back(
          Matrix::with_same_shape(m_hiddenLayers[i]->m_bias));
      weightGradients.emplace_back(
          Matrix::with_same_shape(m_hiddenLayers[i]->m_weights));
    }
    biasGradients.emplace_back(Matrix::with_same_shape(m_outputLayer->m_bias));
    weightGradients.emplace_back(
        Matrix::with_same_shape(m_outputLayer->m_weights));

    Matrix x = xs.get_col(exampleIndex);
    Matrix y = ys.get_col(exampleIndex);

    Matrix out = forward(x);

    Matrix delta =
        CostFunction::get_from_name(m_costFunction)
            .deriv(out, y)
            .elem_multiply_inplace(m_outputLayer->get_activation_fn().deriv(
                m_outputLayer->get_z()));

    biasGradients.back() = delta;
    weightGradients.back() = delta.dot_multiply(
        m_hiddenLayers.back()->get_activations().transpose());

    for (int i = (m_hiddenLayers.size() - 1); i >= 0; i--) {
      Matrix z = m_hiddenLayers[i]->get_z();
      Matrix activationDeriv = m_hiddenLayers[i]->get_activation_fn().deriv(z);

      if (i == m_hiddenLayers.size() - 1) {
        delta = m_outputLayer->m_weights.transpose()
                    .dot_multiply(delta)
                    .elem_multiply_inplace(activationDeriv);
      } else {
        delta = m_hiddenLayers[i + 1]
                    ->m_weights.transpose()
                    .dot_multiply(delta)
                    .elem_multiply_inplace(activationDeriv);
      }

      biasGradients[i] = delta;
      if (i > 0) {
        weightGradients[i] = delta.dot_multiply(
            m_hiddenLayers[i - 1]->get_activations().transpose());
      } else {
        weightGradients[i] =
            delta.dot_multiply(m_inputLayer->get_activations().transpose());
      }
    }

    return std::tuple<std::vector<Matrix>, std::vector<Matrix>>(weightGradients,
                                                                biasGradients);
  }

  void train(const Matrix &trainX, const Matrix &trainY, size_t batchSize,
             size_t epochs, float learningRate) {
    for (size_t e = 0; e < epochs; e++) {
      size_t batchNum = 0;
      for (size_t i = 0; i < trainX.cols(); i += batchSize, batchNum++) {
        update_batch(trainX, trainY, i, std::min(i + batchSize, trainX.cols()),
                     learningRate);

        int correct = 0;
        for (size_t j = i; j < std::min(i + batchSize, trainX.cols()); j++) {
          Matrix out = forward(trainX.get_col(j));
          int correctIdx = 0;
          int maxIdx = 0;
          float maxVal = 0;

          for (size_t k = 0; k < out.rows(); k++) {
            float x = out.get(k, 0);
            if (x > maxVal) {
              maxVal = x;
              maxIdx = k;
            }
          }

          for (size_t k = 0; k < out.rows(); k++) {
            if (trainY.get(k, j) == 1) {
              correctIdx = k;
              break;
            }
          }

          if (correctIdx == maxIdx)
            correct++;
        }

        std::cout << "BATCH " << batchNum << " EPOCH " << e
                  << " \t| ACCURACY: " << ((float)correct / batchSize) << "\n";
      }
    }
  }

  void save(std::filesystem::path outPath) {
    assert(m_inputLayer && m_outputLayer);
    std::ofstream stream(outPath, std::ios::out | std::ios::binary);

    if (!stream.is_open()) {
      std::cerr << "Couldn't open output stream for " << outPath << "\n";
      return;
    }

    std::cout << "Saving model to " << outPath << "\n";

    stream.write("DENDRITE_MODEL\0",
                 15); // All model files will start with this
    uint64_t numLayers = num_layers();
    stream.write((const char *)&numLayers,
                 sizeof(numLayers)); // Number of layers

    stream.write(m_costFunction.c_str(), m_costFunction.size() + 1);
    // stream << m_costFunction << "\0"; // Model's cost function name

    uint64_t numInputs = m_inputLayer->num_inputs();
    stream.write(reinterpret_cast<const char *>(&numInputs),
                 sizeof(numInputs)); // Input layer

    // Iterate over hidden layers
    for (std::shared_ptr<HiddenLayer> hl : m_hiddenLayers) {
      hl->write(stream);
    }
    m_outputLayer->write(stream);

    stream.close();
    std::cout << "Model saved!";
  }

  void load(std::filesystem::path path) {
    assert(!m_inputLayer && !m_outputLayer && m_hiddenLayers.empty());

    std::ifstream stream(path, std::ios::binary);

    std::string fileType;
    std::getline(stream, fileType, '\0');

    if (fileType != "DENDRITE_MODEL") {
      std::cerr << "Invalid model file type for " << path << "!\n";
      return;
    }

    uint64_t numLayers = 0;
    stream.read(reinterpret_cast<char *>(&numLayers), sizeof(numLayers));

    std::cout << "num layers: " << numLayers << "\n";

    std::string costFn;
    std::getline(stream, costFn, '\0');

    m_costFunction = costFn;

    uint64_t numInputs;
    stream.read(reinterpret_cast<char *>(&numInputs), sizeof(numInputs));
    std::cout << "cost fn: " << costFn << "\n";

    set_input_layer(numInputs);

    for (size_t i = 0; i < numLayers - 2; i++) {
      std::shared_ptr<Layer> prev;
      if (i == 0) {
        prev = std::shared_ptr<Layer>(m_inputLayer);
      } else {
        prev = m_hiddenLayers[m_hiddenLayers.size() - 1];
      }

      HiddenLayer hl = HiddenLayer::load(stream, prev);

      m_hiddenLayers.emplace_back(std::make_shared<HiddenLayer>(hl));
    }
    m_outputLayer = std::make_shared<OutputLayer>(
        OutputLayer::load(stream, m_hiddenLayers.back()));

    stream.close();
  }

  size_t num_layers() const {
    int n = 0;
    if (m_inputLayer)
      n++;
    if (m_outputLayer)
      n++;
    n += m_hiddenLayers.size();
    return n;
  }
};

#endif

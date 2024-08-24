#ifndef NN_H
#define NN_H

#include "math/ActivationFunction.hpp"
#include "math/CostFunction.hpp"
#include "math/Matrix.hpp"
#include "nn/Layer.hpp"
#include <cassert>
#include <memory>

class NeuralNetwork {
private:
  std::vector<std::shared_ptr<HiddenLayer>> m_hiddenLayers;
  std::shared_ptr<OutputLayer> m_outputLayer;
  std::shared_ptr<InputLayer> m_inputLayer;
  const CostFunction &m_costFunction;

public:
  NeuralNetwork(const NeuralNetwork &other)
      : m_costFunction(other.m_costFunction) {}
  NeuralNetwork(const CostFunction &costFn) : m_costFunction(costFn) {}

  void set_input_layer(int numInputs) {
    this->m_inputLayer = std::make_shared<InputLayer>(numInputs);
  }

  void add_hidden_layer(int numNeurons, const ActivationFunction &fn) {
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

  void set_output_layer(int numOutputs, const ActivationFunction &fn) {
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

    Matrix delta = m_costFunction.deriv(out, y).elem_multiply_inplace(
        m_outputLayer->get_activation_fn().deriv(m_outputLayer->get_z()));

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

  void train(const Matrix &trainX, const Matrix &trainY, size_t batchSize) {
    for (size_t e = 0; e < 1000; e++) {
      size_t batchNum = 0;
      for (size_t i = 0; i < trainX.cols(); i += batchSize, batchNum++) {
        update_batch(trainX, trainY, i, std::min(i + batchSize, trainX.cols()),
                     0.1);

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

  int num_layers() const {
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

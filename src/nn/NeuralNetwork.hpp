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

    m_hiddenLayers.emplace_back(
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
    for (int i = 0; i < m_hiddenLayers.size(); i++) {
      m_hiddenLayers[i]->rand_init();
    }
    m_outputLayer->rand_init();
  }

  Matrix forward(const Matrix &inputs) {
    m_inputLayer->set_inputs(inputs);
    int i = 0;
    for (int i = 0; i < m_hiddenLayers.size(); i++) {
      // std::cout << "=====Calculating Activations For Layer " << i << "\n";
      m_hiddenLayers[i]->calc_activations();
    }
    return m_outputLayer->calc_outputs();
  }

  void
  update_batch(Matrix x,
               Matrix y) { // Columns in x and y should be inputs/output vectors
    assert(x.cols() == y.cols());
    assert(m_inputLayer);
    assert(m_outputLayer);

    std::vector<Matrix> layerWeightGradients;
    std::vector<Matrix> layerBiasGradients;

    for (int i = 0; i < m_hiddenLayers.size(); i++) {
      layerBiasGradients.emplace_back(
          Matrix(m_hiddenLayers[i]->num_neurons(), 1));
      layerWeightGradients.emplace_back(
          Matrix(m_hiddenLayers[i]->num_neurons(),
                 m_hiddenLayers[i]->get_prev_layer()->num_neurons()));
    }
    layerBiasGradients.emplace_back(Matrix(m_outputLayer->num_neurons(), 1));
    layerWeightGradients.emplace_back(
        Matrix(m_outputLayer->num_neurons(),
               m_outputLayer->get_prev_layer()->num_neurons()));
  }

  std::tuple<Matrix, Matrix>
  backprop(Matrix x,
           Matrix y) { // Columns in x and y should be inputs/output vectors
    assert(x.cols() == y.cols());
    return std::tuple<Matrix, Matrix>(Matrix(1, 1), Matrix(1, 1));
  }

  void train(Matrix trainX, Matrix trainY, int epochs, float iterations) {}

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

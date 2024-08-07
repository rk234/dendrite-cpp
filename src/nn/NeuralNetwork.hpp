#ifndef NN_H
#define NN_H

#include "math/ActivationFunction.hpp"
#include "math/CostFunction.hpp"
#include "math/Matrix.hpp"
#include "nn/Layer.hpp"
#include <memory>

class NeuralNetwork {
private:
  std::vector<std::shared_ptr<HiddenLayer>> m_hiddenLayers;
  std::shared_ptr<OutputLayer> m_outputLayer;
  std::shared_ptr<InputLayer> m_inputLayer;
  CostFunction &m_costFunction;

public:
  NeuralNetwork(CostFunction &costFn) : m_costFunction(costFn) {}

  void set_input_layer(int numInputs) {
    this->m_inputLayer = std::make_shared<InputLayer>(numInputs);
  }

  void add_hidden_layer(int numNeurons, const ActivationFunction &fn) {
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

  void train(Matrix trainX, Matrix trainY, int epochs, float iterations) {}
};

#endif

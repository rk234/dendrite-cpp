#ifndef NN_H
#define NN_H

#include "math/ActivationFunction.hpp"
#include "math/Matrix.hpp"
#include "math/ReLU.hpp"
#include "nn/Layer.hpp"

class NeuralNetwork {
private:
  std::vector<HiddenLayer> m_hiddenLayers;
  OutputLayer *m_outputLayer;
  InputLayer *m_inputLayer;

public:
  ~NeuralNetwork() {
    delete m_inputLayer;
    delete m_outputLayer;
  }

  void set_input_layer(int numInputs) {
    this->m_inputLayer = new InputLayer(numInputs);
  }

  void add_hidden_layer(int numNeurons, const ActivationFunction &fn) {
    Layer *prev;

    if (m_hiddenLayers.size() > 0) {
      prev = &m_hiddenLayers.back();
    } else {
      prev = (Layer *)m_inputLayer;
    }

    m_hiddenLayers.push_back(HiddenLayer(numNeurons, prev, fn));
  }

  void set_output_layer(int numOutputs, const ActivationFunction &fn) {
    Layer *prev;

    if (m_hiddenLayers.size() > 0) {
      prev = &m_hiddenLayers.back();
    } else {
      prev = (Layer *)m_inputLayer;
    }

    this->m_outputLayer = new OutputLayer(numOutputs, prev, fn);
  }

  void init() {
    for (HiddenLayer l : m_hiddenLayers) {
      l.rand_init();
    }
    m_outputLayer->rand_init();
  }

  Matrix forward(const Matrix &inputs) {
    m_inputLayer->setInputs(inputs);
    for (HiddenLayer l : m_hiddenLayers) {
      l.calc_activations();
    }
    return m_outputLayer->calc_outputs();
  }

  void train(Matrix trainX, Matrix trainY, int epochs,
             float iterations); // TODO
};

#endif

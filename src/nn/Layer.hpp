#ifndef LAYER_H
#define LAYER_H

#include "math/Matrix.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <random>
class Layer {
protected:
  int m_neurons;
  Matrix m_activations; // Single column, Rows are activations of this layer's
                        // perceptrons
public:
  Layer(int numNeurons) : m_activations(numNeurons, 1) {
    m_neurons = numNeurons;
  }

  Layer *set_activations(const Matrix &activations) {
    m_activations.set_data_from(activations);
    std::cout << "Input Activations:\n";
    m_activations.print();
    return this;
  }

  const Matrix &get_activations() { return m_activations; }
};

class InputLayer : public Layer {
public:
  InputLayer(int numInputs) : Layer(numInputs) {}

  Layer *set_inputs(const Matrix &inputs) {
    assert(inputs.cols() == 1 && inputs.rows() == m_neurons);
    return set_activations(inputs);
  }
};

class HiddenLayer : public Layer {
protected:
  Layer *m_prevLayer;
  Matrix m_weights;
  Matrix m_bias;
  const ActivationFunction &m_fn;

public:
  HiddenLayer(const HiddenLayer &other)
      : Layer(other.m_weights.rows()), m_weights(other.m_weights),
        m_bias(other.m_bias), m_fn(other.m_fn) {
    m_prevLayer = other.m_prevLayer;
  }

  HiddenLayer(int numNeurons, Layer *prevLayer, const ActivationFunction &fn)
      : Layer(numNeurons),
        m_weights(numNeurons, prevLayer->get_activations().rows()),
        m_bias(numNeurons, 1), m_fn(fn) {
    m_prevLayer = prevLayer;
  }

  Matrix &calc_activations() {
    std::cout << "HERE" << std::endl;
    Matrix activations = m_weights * m_prevLayer->get_activations();
    std::cout << "WEIGHTS and BIAS";
    m_weights.print();
    m_bias.print();
    std::cout << std::endl;

    activations.add_inplace(m_bias);

    activations.apply_activation_inplace(m_fn);
    m_activations = activations;

    return m_activations;
  }

  void rand_init() {
    std::normal_distribution<float> dist;
    std::default_random_engine generator;
    generator.seed(std::random_device{}());

    std::cout << "BIAS: ";
    for (int i = 0; i < m_neurons; i++) {
      m_bias.set(i, 0, dist(generator));
    }
    m_bias.print();
    std::cout << "\n";

    const Matrix &prevActivations = m_prevLayer->get_activations();
    for (int i = 0; i < m_activations.rows(); i++) {
      for (int j = 0; j < prevActivations.rows(); j++) {
        m_weights.set(i, j, dist(generator));
      }
    }
  }
};

class OutputLayer : public HiddenLayer {
public:
  OutputLayer(int numOutputs, Layer *prevLayer, const ActivationFunction &fn)
      : HiddenLayer(numOutputs, prevLayer, fn) {}

  Matrix &calc_outputs() { return calc_activations(); }

  Matrix &getOutputs() { return m_activations; }
};

#endif // !LAYER_H

#ifndef LAYER_H
#define LAYER_H

#include "math/Matrix.hpp"
#include <cassert>
#include <cstdlib>
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
    return this;
  }

  Matrix get_activations() { return m_activations; }
};

class InputLayer : Layer {
public:
  InputLayer(int numInputs) : Layer(numInputs) {}

  Layer *setInputs(const Matrix &inputs) {
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
    Matrix activations = m_weights * m_prevLayer->get_activations();
    activations.apply_activation_inplace(m_fn);
    m_activations = activations;
    return m_activations;
  }

  void rand_init() {
    for (int i = 0; i < m_neurons; i++) {
      // TODO: This should init weights lol, wasn't fully awake writing this
      //  m_activations.set(i, 0, ((float)rand()) / RAND_MAX);
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

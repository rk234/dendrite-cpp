#ifndef LAYER_H
#define LAYER_H

#include "math/ActivationFunction.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <cstdlib>
#include <memory>
#include <random>

class Layer {
protected:
  size_t m_neurons;
  Matrix m_activations; // Single column, Rows are activations of this layer's
                        // perceptrons
public:
  Layer(size_t numNeurons) : m_activations(numNeurons, 1) {
    m_neurons = numNeurons;
  }

  Layer *set_activations(const Matrix &activations) {
    m_activations.set_data_from(activations);
    // std::cout << "Input Activations:\n";
    // m_activations.print();
    return this;
  }

  const Matrix &get_activations() const { return m_activations; }
  int num_neurons() const { return m_neurons; }
};

class InputLayer : public Layer {
public:
  InputLayer(size_t numInputs) : Layer(numInputs) {}

  Layer *set_inputs(const Matrix &inputs) {
    assert(inputs.cols() == 1 && inputs.rows() == m_neurons);
    return set_activations(inputs);
  }

  int num_inputs() const { return m_neurons; }
};

class HiddenLayer : public Layer {
protected:
  std::shared_ptr<Layer> m_prevLayer;
  Matrix m_z;
  const ActivationFunction &m_fn;

public:
  Matrix m_weights; // This layer's neurons x Previous layer's neurones
  Matrix m_bias;

  HiddenLayer(const HiddenLayer &other)
      : Layer(other.m_weights.rows()),
        m_z(other.m_activations.rows(), other.m_activations.cols()),
        m_fn(other.m_fn), m_weights(other.m_weights), m_bias(other.m_bias) {
    m_prevLayer = other.m_prevLayer;
  }

  HiddenLayer(size_t numNeurons, std::shared_ptr<Layer> prevLayer,
              const ActivationFunction &fn)
      : Layer(numNeurons), m_z(m_activations.rows(), m_activations.cols()),
        m_fn(fn), m_weights(numNeurons, prevLayer->get_activations().rows()),
        m_bias(numNeurons, 1) {
    m_prevLayer = prevLayer;
  }

  const Matrix &get_z() const { return m_z; }

  Matrix &calc_activations() {
    m_z = (m_weights * m_prevLayer->get_activations()).add_inplace(m_bias);
    m_activations = m_fn.activate(m_z);
    return m_activations;
  }

  void rand_init() {
    std::normal_distribution<float> dist;
    std::default_random_engine generator;
    generator.seed(std::random_device{}());

    for (size_t i = 0; i < m_neurons; i++) {
      m_bias.set(i, 0, dist(generator));
    }

    const Matrix &prevActivations = m_prevLayer->get_activations();
    for (size_t i = 0; i < m_activations.rows(); i++) {
      for (size_t j = 0; j < prevActivations.rows(); j++) {
        m_weights.set(i, j, dist(generator));
      }
    }
  }

  std::shared_ptr<Layer> get_prev_layer() { return m_prevLayer; }
};

class OutputLayer : public HiddenLayer {
public:
  OutputLayer(int numOutputs, std::shared_ptr<Layer> prevLayer,
              const ActivationFunction &fn)
      : HiddenLayer(numOutputs, prevLayer, fn) {}

  const Matrix &calc_outputs() { return calc_activations(); }
  const Matrix &get_outputs() const { return m_activations; }
};

#endif // !LAYER_H

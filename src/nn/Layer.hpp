#ifndef LAYER_H
#define LAYER_H

#include "math/ActivationFunction.hpp"
#include "math/Matrix.hpp"
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <memory>

namespace Dendrite {
class Layer {
protected:
  size_t m_neurons;
  Matrix m_activations; // Single column, Rows are activations of this layer's
                        // perceptrons
public:
  Layer(size_t numNeurons) : m_activations(numNeurons, 1) {
    m_neurons = numNeurons;
  }

  Layer *set_activations(const Matrix &activations);

  const Matrix &get_activations() const { return m_activations; }
  int num_neurons() const { return m_neurons; }
};

class InputLayer : public Layer {
public:
  InputLayer(size_t numInputs) : Layer(numInputs) {}

  Layer *set_inputs(const Matrix &inputs);

  int num_inputs() const { return m_neurons; }
};

class HiddenLayer : public Layer {
protected:
  Matrix m_z;
  std::string m_fn;

public:
  std::shared_ptr<Layer> m_prevLayer;
  Matrix m_weights; // This layer's neurons x Previous layer's neurones
  Matrix m_bias;

  HiddenLayer(const HiddenLayer &other)
      : Layer(other.m_weights.rows()),
        m_z(other.m_activations.rows(), other.m_activations.cols()),
        m_fn(other.m_fn), m_weights(other.m_weights), m_bias(other.m_bias) {
    m_prevLayer = other.m_prevLayer;
  }

  HiddenLayer(size_t numNeurons, std::shared_ptr<Layer> prevLayer,
              std::string fn)
      : Layer(numNeurons), m_z(m_activations.rows(), m_activations.cols()),
        m_fn(fn), m_weights(numNeurons, prevLayer->get_activations().rows()),
        m_bias(numNeurons, 1) {
    m_prevLayer = prevLayer;
  }

  const Matrix &get_z() const { return m_z; }
  const ActivationFunction &get_activation_fn() const {
    return ActivationFunction::get_from_name(m_fn);
  }
  const std::string &get_activation_fn_name() const { return m_fn; }

  Matrix &calc_activations();

  void rand_init();

  void write(std::basic_ofstream<char> &stream);

  static HiddenLayer load(std::basic_ifstream<char> &stream,
                          std::shared_ptr<Layer> prevLayer);

  std::shared_ptr<Layer> get_prev_layer() { return m_prevLayer; }
};

class OutputLayer : public HiddenLayer {
public:
  OutputLayer(int numOutputs, std::shared_ptr<Layer> prevLayer,
              const std::string &fn)
      : HiddenLayer(numOutputs, prevLayer, fn) {}

  const Matrix &calc_outputs() { return calc_activations(); }
  const Matrix &get_outputs() const { return m_activations; }

  static OutputLayer load(std::basic_ifstream<char> &stream,
                          std::shared_ptr<Layer> prevLayer);
};
} // namespace Dendrite

#endif // !LAYER_H

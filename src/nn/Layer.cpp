#include "Layer.hpp"
#include <random>

namespace Dendrite {
Layer *Layer::set_activations(const Matrix &activations) {
  m_activations.set_data_from(activations);
  // std::cout << "Input Activations:\n";
  // m_activations.print();
  return this;
}

Layer *InputLayer::set_inputs(const Matrix &inputs) {
  assert(inputs.cols() == 1 && inputs.rows() == m_neurons);
  return set_activations(inputs);
}

Matrix &HiddenLayer::calc_activations() {
  m_z = (m_weights * m_prevLayer->get_activations()).add_inplace(m_bias);
  m_activations = ActivationFunction::get_from_name(m_fn).activate(m_z);
  return m_activations;
}

void HiddenLayer::rand_init() {
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

void HiddenLayer::write(std::basic_ofstream<char> &stream) {
  uint64_t numNeurons = num_neurons();
  stream.write(reinterpret_cast<const char *>(&numNeurons), // Number of neurons
               sizeof(numNeurons));

  stream.write(get_activation_fn_name().c_str(),
               get_activation_fn_name().size() + 1);
  // stream << get_activation_fn_name() << "\0"; // Activation function

  uint64_t weightRows = m_weights.rows();
  uint64_t weightCols = m_weights.cols();

  stream.write(reinterpret_cast<const char *>(&weightRows),
               sizeof(weightRows)); // Input layer
  stream.write(reinterpret_cast<const char *>(&weightCols),
               sizeof(weightCols)); // Input layer

  for (size_t i = 0; i < weightRows * weightCols; i++) {
    float w = m_weights.get_data()[i];
    stream.write(reinterpret_cast<const char *>(&w), sizeof(w));
  }

  for (size_t i = 0; i < numNeurons; i++) {
    float b = m_bias.get_data()[i];
    stream.write(reinterpret_cast<const char *>(&b), sizeof(b));
  }
}

HiddenLayer HiddenLayer::load(std::basic_ifstream<char> &stream,
                              std::shared_ptr<Layer> prevLayer) {
  uint64_t numNeurons;
  stream.read(reinterpret_cast<char *>(&numNeurons), sizeof(numNeurons));

  std::string activationFn;
  std::getline(stream, activationFn, '\0');

  // std::cout << activationFn << "\n";
  uint64_t weightRows;
  uint64_t weightCols;

  stream.read(reinterpret_cast<char *>(&weightRows), sizeof(weightRows));
  stream.read(reinterpret_cast<char *>(&weightCols), sizeof(weightCols));

  Matrix weights = Matrix(weightRows, weightCols);
  Matrix biases = Matrix(numNeurons, 1);

  for (size_t i = 0; i < weightRows * weightCols; i++) {
    float w;
    stream.read(reinterpret_cast<char *>(&w), sizeof(w));
    weights.set_data(i, w);
  }
  for (size_t i = 0; i < numNeurons; i++) {
    float b;
    stream.read(reinterpret_cast<char *>(&b), sizeof(b));
    biases.set_data(i, b);
  }

  HiddenLayer out = HiddenLayer(numNeurons, prevLayer, activationFn);
  out.m_weights = weights;
  out.m_bias = biases;

  return out;
}

OutputLayer OutputLayer::load(std::basic_ifstream<char> &stream,
                              std::shared_ptr<Layer> prevLayer) {
  HiddenLayer layer = HiddenLayer::load(stream, prevLayer);
  return *static_cast<OutputLayer *>(&layer);
}

} // namespace Dendrite

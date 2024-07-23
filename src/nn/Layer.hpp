#ifndef LAYER_H
#define LAYER_H

#include "math/ColumnVector.hpp"
#include "math/Matrix.hpp"
class Layer {
private:
  Matrix m_weightsFromPrev; // Rows => perceptrons in this layer, Columns =>
                            // weights for previous layer's activations
  ColumnVector
      m_activations;   // Rows are activations of this layer's perceptrons
  ColumnVector m_bias; // Rows => perceptrons in this layer
  Layer *m_prevLayer;
  Layer *m_nextLayer;

public:
  Layer(int numPerceptrons)
      : m_bias(numPerceptrons), m_activations(numPerceptrons),
        m_weightsFromPrev(1, 1, 0) {
    m_prevLayer = nullptr;
    m_nextLayer = nullptr;
  }

  Layer(int numPerceptrons, Layer *prev)
      : m_bias(numPerceptrons), m_activations(numPerceptrons),
        m_weightsFromPrev(numPerceptrons, prev->getActivations()->len()) {
    assert(prev != nullptr);
    m_prevLayer = prev;
    m_nextLayer = nullptr;
    m_prevLayer->m_nextLayer = this;
  }

  ColumnVector *getActivations() { return &m_activations; }
};

#endif // !LAYER_H

#ifndef NN_H
#define NN_H

#include "math/Matrix.hpp"
class NeuralNetwork {
private:
  int m_numInputs;
  int m_numOutputs;

public:
  void add_input_layer(int numInputs);
  void add_hidden_layer(int numPerceptrons);
  void add_output_layer(int numOutputs);

  Matrix *forward(Matrix &inputs);
};

#endif

#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
class Matrix {
private:
  int rows;
  int cols;
  std::vector<float> elements;

public:
  Matrix(int rows, int cols) {}

  Matrix() {}
};
#endif // !MATRIX_H

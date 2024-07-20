#ifndef MATRIX_H
#define MATRIX_H
#include <cassert>
#include <vector>
class Matrix {
private:
  int m_rows;
  int m_cols;
  std::vector<float> m_elements;

public:
  Matrix(int rows, int cols) {
    this->m_rows = rows;
    this->m_cols = cols;
    this->m_elements = std::vector<float>(rows * cols, 0.0f);
  }

  Matrix(Matrix &mat) {
    this->m_rows = mat.m_rows;
    this->m_cols = mat.m_cols;
    this->m_elements = std::vector<float>(mat.get_elements());
  }

  float get(int i, int j) { return m_elements[i * m_cols + j]; }
  void set(int i, int j, float val) { m_elements[i * m_cols + j] = val; }
  std::vector<float> &get_elements() { return this->m_elements; }

  int rows() { return m_rows; }

  int cols() { return m_cols; }

  Matrix multiply(Matrix other) {
    Matrix res = Matrix(m_rows, other.cols());

    return res;
  }

  Matrix operator*(Matrix other) {
    assert(m_cols == other.rows());

    Matrix res = Matrix(m_rows, other.cols());

    for (int row = 0; row < m_rows; row++) {
      for (int col = 0; col < m_cols; col++) {
      }
    }

    return res;
  }
};
#endif // !MATRIX_H

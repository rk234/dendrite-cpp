#ifndef MATRIX_H
#define MATRIX_H
#include <cassert>
#include <iostream>
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

  Matrix(int rows, int cols, float fillVal) {
    this->m_rows = rows;
    this->m_cols = cols;
    this->m_elements = std::vector<float>(rows * cols, fillVal);
  }

  Matrix(Matrix &mat) {
    this->m_rows = mat.m_rows;
    this->m_cols = mat.m_cols;
    this->m_elements = std::vector<float>(mat.get_elements());
  }

  Matrix(float (&data)[], int rows, int cols) {
    this->m_rows = rows;
    this->m_cols = cols;
    this->m_elements = std::vector<float>(rows * cols);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->m_elements[i * m_cols + j] = (data[i * cols + j]);
      }
    }
  }

  template <int rows, int cols> Matrix(float (&data)[rows][cols]) {
    this->m_rows = rows;
    this->m_cols = cols;
    this->m_elements = std::vector<float>(rows * cols);

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->m_elements[i * m_cols + j] = (data[i][j]);
      }
    }
  }

  float get(int i, int j) const { return m_elements[i * m_cols + j]; }

  // could look into optimizing this by returning reference,
  // but wouldn't be consistent with current get_col implementation
  std::vector<float> get_row(float i) const {
    std::vector<float> row = std::vector<float>(m_cols);

    for (int j = 0; j < m_cols; j++) {
      row.push_back(get(i, j));
    }

    return row;
  }

  std::vector<float> get_col(float j) const {
    std::vector<float> col = std::vector<float>(m_rows);

    for (int i = 0; i < m_rows; i++) {
      col.push_back(get(i, j));
    }

    return col;
  }

  void set(int i, int j, float val) { m_elements[i * m_cols + j] = val; }

  void set_data(std::vector<float> &data) { m_elements.swap(data); }

  std::vector<float> &get_elements() { return this->m_elements; }

  int rows() const { return m_rows; }

  int cols() const { return m_cols; }

  Matrix multiply(Matrix other) const {
    assert(m_cols == other.rows());

    Matrix res = Matrix(m_rows, other.cols());

    for (int r = 0; r < m_rows; r++) {
      for (int c = 0; c < other.cols(); c++) {
        float sum = 0;
        for (int i = 0; i < other.rows(); i++)
          sum += other.get(i, c) * get(r, i);
        res.set(r, c, sum);
      }
    }

    return res;
  }

  Matrix operator*(Matrix other) const { return multiply(other); }

  void print() const {
    for (int i = 0; i < m_rows; i++) {
      std::cout << "| ";
      for (int j = 0; j < m_cols; j++) {
        std::cout << get(i, j) << " ";
      }
      std::cout << "|";
      std::cout << "\n";
    }
  }
};
#endif // !MATRIX_H

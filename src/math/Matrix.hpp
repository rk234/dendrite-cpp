#ifndef MATRIX_H
#define MATRIX_H
#include "math/ActivationFunction.hpp"
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

  Matrix(const Matrix &mat) {
    this->m_rows = mat.m_rows;
    this->m_cols = mat.m_cols;
    this->m_elements = std::vector<float>(mat.get_data());
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

  void set_data(std::vector<float> data) {
    assert(m_rows * m_cols == data.size());
    m_elements = data;
  }

  void set_data_from(const Matrix &mat) {
    assert(same_shape(mat));
    set_data(mat.get_data());
  }

  const std::vector<float> &get_data() const { return this->m_elements; }

  int rows() const { return m_rows; }

  int cols() const { return m_cols; }

  Matrix multiply(const Matrix &other) const {
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

  Matrix operator*(const Matrix &other) const { return multiply(other); }
  Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      m_elements = other.m_elements;
      m_cols = other.m_cols;
      m_rows = other.m_rows;
    }

    return *this;
  }

  Matrix add(float x) const {
    Matrix out = Matrix(m_rows, m_cols);
    for (int i = 0; i < m_rows; i++) {
      for (int j = 0; j < m_cols; j++) {
        out.set(i, j, get(i, j) + x);
      }
    }
    return out;
  }

  Matrix *add_inplace(float x) {
    for (int i = 0; i < m_rows; i++) {
      for (int j = 0; j < m_cols; j++) {
        m_elements[i * m_cols + j] += x;
      }
    }

    return this;
  }

  Matrix add(const Matrix &other) {
    assert(other.rows() == m_rows && other.cols() == m_cols);
    Matrix res = Matrix(m_rows, m_cols);

    for (int i = 0; i < m_rows; i++) {
      for (int j = 0; j < m_rows; j++) {
        res.set(i, j, get(i, j) + other.get(i, j));
      }
    }

    return res;
  }

  Matrix *add_inplace(const Matrix &other) {
    assert(other.rows() == m_rows && other.cols() == m_cols);
    for (int i = 0; i < m_rows; i++) {
      for (int j = 0; j < m_rows; j++) {
        set(i, j, get(i, j) + other.get(i, j));
      }
    }
    return this;
  }

  Matrix apply_activation(const ActivationFunction &func) const {
    Matrix out = Matrix(m_rows, m_cols);
    for (int i = 0; i < m_rows; i++) {
      for (int j = 0; j < m_cols; j++) {
        out.set(i, j, func.activate(get(i, j)));
      }
    }
    return out;
  }

  Matrix *apply_activation_inplace(const ActivationFunction &func) {
    for (int i = 0; i < m_rows; i++) {
      for (int j = 0; j < m_cols; j++) {
        set(i, j, func.activate(get(i, j)));
      }
    }
    return this;
  }

  bool same_shape(const Matrix &other) const {
    return (m_cols == other.cols() && m_rows == other.rows());
  }

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

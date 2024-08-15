#ifndef MATRIX_H
#define MATRIX_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>
class Matrix {
private:
  size_t m_rows;
  size_t m_cols;
  std::vector<float> m_elements;

public:
  Matrix() {
    m_rows = 0;
    m_cols = 0;
    m_elements = std::vector<float>();
  }

  Matrix(size_t rows, size_t cols) {
    this->m_rows = rows;
    this->m_cols = cols;
    this->m_elements = std::vector<float>(rows * cols, 0.0f);
  }

  Matrix(size_t rows, size_t cols, float fillVal) {
    this->m_rows = rows;
    this->m_cols = cols;
    this->m_elements = std::vector<float>(rows * cols, fillVal);
  }

  Matrix(const Matrix &mat) {
    this->m_rows = mat.m_rows;
    this->m_cols = mat.m_cols;
    this->m_elements = std::vector<float>(mat.get_data());
  }

  Matrix(float (&data)[], size_t rows, size_t cols) {
    this->m_rows = rows;
    this->m_cols = cols;
    this->m_elements = std::vector<float>(rows * cols);

    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
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

  float get(size_t i, size_t j) const {
    assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols);
    return m_elements[i * m_cols + j];
  }

  // could look into optimizing this by returning reference,
  // but wouldn't be consistent with current get_col implementation
  Matrix get_row(size_t i) const {
    Matrix row = Matrix(1, m_cols);

    for (size_t j = 0; j < m_cols; j++) {
      row.set(0, j, get(i, j));
    }

    return row;
  }

  Matrix get_col(size_t j) const {
    Matrix col = Matrix(m_rows, 1);
    for (size_t i = 0; i < m_rows; i++) {
      col.set(i, 0, get(i, j));
    }

    return col;
  }

  void set(size_t i, size_t j, float val) {
    assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols);
    m_elements[i * m_cols + j] = val;
  }

  void set_data(std::vector<float> data) {
    assert(m_rows * m_cols == data.size());
    m_elements = data;
  }

  void set_data(size_t i, float f) {
    assert(i >= 0 && i < m_elements.size());
    m_elements[i] = f;
  }

  void set_data_from(const Matrix &mat) {
    assert(same_shape(mat));
    set_data(mat.get_data());
  }

  const std::vector<float> &get_data() const { return this->m_elements; }

  size_t rows() const { return m_rows; }

  size_t cols() const { return m_cols; }

  Matrix &scale_inplace(float s) {
    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        set(i, j, get(i, j) * s);
      }
    }
    return *this;
  }

  Matrix scale(float s) const {
    Matrix out = Matrix(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        out.set(i, j, get(i, j) * s);
      }
    }
    return out;
  }

  Matrix dot_multiply(const Matrix &other) const {
    assert(m_cols == other.rows());

    Matrix res = Matrix(m_rows, other.cols());

    for (size_t r = 0; r < m_rows; r++) {
      for (size_t c = 0; c < other.cols(); c++) {
        float sum = 0;
        for (size_t i = 0; i < other.rows(); i++)
          sum += other.get(i, c) * get(r, i);
        res.set(r, c, sum);
      }
    }

    return res;
  }

  Matrix elem_multiply(const Matrix &other) const {
    assert(same_shape(other));

    Matrix out = Matrix(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        out.set(i, j, get(i, j) * other.get(i, j));
      }
    }

    return out;
  }

  Matrix &elem_multiply_inplace(const Matrix &other) {
    assert(same_shape(other));

    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        set(i, j, get(i, j) * other.get(i, j));
      }
    }

    return *this;
  }

  Matrix add(float x) const {
    Matrix out = Matrix(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        out.set(i, j, get(i, j) + x);
      }
    }
    return out;
  }

  Matrix &add_inplace(float x) {
    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        m_elements[i * m_cols + j] += x;
      }
    }

    return *this;
  }

  Matrix add(const Matrix &other) const {
    assert(other.rows() == m_rows && other.cols() == m_cols);
    Matrix res = Matrix(m_rows, m_cols);

    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        res.set(i, j, get(i, j) + other.get(i, j));
      }
    }

    return res;
  }

  Matrix &add_inplace(const Matrix &other) {
    assert(other.rows() == m_rows && other.cols() == m_cols);
    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        set(i, j, get(i, j) + other.get(i, j));
      }
    }
    return *this;
  }

  Matrix pow_elem(float p) const {
    return apply_function([p](float x) -> float { return std::pow(x, p); });
  }

  Matrix &pow_elem_inplace(float p) {
    return apply_function_inplace(
        [p](float x) -> float { return std::pow(x, p); });
  }

  Matrix transpose() const {
    Matrix t = Matrix(m_cols, m_rows);

    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        t.set(j, i, get(i, j));
      }
    }

    return t;
  }

  Matrix operator*(const Matrix &other) const { return dot_multiply(other); }
  Matrix operator*(float f) const { return scale(f); }
  friend Matrix operator*(float f, const Matrix &other) { return other * f; }

  Matrix operator+(const Matrix &other) const { return add(other); }
  const Matrix &operator+=(const Matrix &other) { return add_inplace(other); }
  const Matrix &operator-=(const Matrix &other) {
    return add_inplace(other.scale(-1.0f));
  }
  Matrix operator-(const Matrix &other) const {
    return add(other.scale(-1.0f));
  }

  Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      m_elements = other.m_elements;
      m_cols = other.m_cols;
      m_rows = other.m_rows;
    }

    return *this;
  }

  Matrix apply_function(std::function<float(float)> func) const {
    Matrix out = Matrix(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        out.set(i, j, func(get(i, j)));
      }
    }
    return out;
  }

  Matrix &apply_function_inplace(std::function<float(float)> func) {
    for (size_t i = 0; i < m_rows; i++) {
      for (size_t j = 0; j < m_cols; j++) {
        set(i, j, func(get(i, j)));
      }
    }
    return *this;
  }

  bool same_shape(const Matrix &other) const {
    return (m_cols == other.cols() && m_rows == other.rows());
  }

  void print() const {
    for (size_t i = 0; i < m_rows; i++) {
      std::cout << "| ";
      for (size_t j = 0; j < m_cols; j++) {
        std::cout << get(i, j) << " ";
      }
      std::cout << "|";
      std::cout << "\n";
    }
  }

  static Matrix with_same_shape(
      const Matrix &other) { // creates matrix with same shape as other
    return Matrix(other.rows(), other.cols());
  }
};

#endif // !MATRIX_H

#ifndef MATRIX_H
#define MATRIX_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace Dendrite {
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

  float get(size_t i, size_t j) const;

  // could look into optimizing this by returning reference,
  // but wouldn't be consistent with current get_col implementation
  Matrix get_row(size_t i) const;

  Matrix get_col(size_t j) const;

  void set(size_t i, size_t j, float val);

  void set_data(std::vector<float> data);

  void set_data(size_t i, float f);

  void set_data_from(const Matrix &mat);

  const std::vector<float> &get_data() const { return this->m_elements; }

  size_t rows() const { return m_rows; }

  size_t cols() const { return m_cols; }

  Matrix &scale_inplace(float s);

  Matrix scale(float s) const;

  Matrix dot_multiply(const Matrix &other) const;

  Matrix elem_multiply(const Matrix &other) const;

  Matrix &elem_multiply_inplace(const Matrix &other);

  Matrix add(float x) const;

  Matrix &add_inplace(float x);

  Matrix add(const Matrix &other) const;

  Matrix &add_inplace(const Matrix &other);

  Matrix pow_elem(float p) const;

  Matrix &pow_elem_inplace(float p);

  Matrix transpose() const;

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

  Matrix apply_function(std::function<float(float)> func) const;

  Matrix &apply_function_inplace(std::function<float(float)> func);

  bool same_shape(const Matrix &other) const;

  void print() const;

  static Matrix
  with_same_shape(const Matrix &other); // creates matrix with same shape
};
} // namespace Dendrite

#endif // !MATRIX_H

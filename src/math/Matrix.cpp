#include "Matrix.hpp"
#include <functional>
#include <iostream>
namespace Dendrite {

float Matrix::get(size_t i, size_t j) const {
  assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols);
  return m_elements[i * m_cols + j];
}

Matrix Matrix::get_row(size_t i) const {
  Matrix row = Matrix(1, m_cols);

  for (size_t j = 0; j < m_cols; j++) {
    row.set(0, j, get(i, j));
  }

  return row;
}

Matrix Matrix::get_col(size_t j) const {
  Matrix col = Matrix(m_rows, 1);
  for (size_t i = 0; i < m_rows; i++) {
    col.set(i, 0, get(i, j));
  }

  return col;
}

void Matrix::set(size_t i, size_t j, float val) {
  assert(i >= 0 && i < m_rows && j >= 0 && j < m_cols);
  m_elements[i * m_cols + j] = val;
}

void Matrix::set_data(std::vector<float> data) {
  assert(m_rows * m_cols == data.size());
  m_elements = data;
}

void Matrix::set_data(size_t i, float f) {
  assert(i >= 0 && i < m_elements.size());
  m_elements[i] = f;
}

void Matrix::set_data_from(const Matrix &mat) {
  assert(same_shape(mat));
  set_data(mat.get_data());
}

Matrix &Matrix::scale_inplace(float s) {
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      set(i, j, get(i, j) * s);
    }
  }
  return *this;
}

Matrix Matrix::scale(float s) const {
  Matrix out = Matrix(m_rows, m_cols);
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      out.set(i, j, get(i, j) * s);
    }
  }
  return out;
}

Matrix Matrix::dot_multiply(const Matrix &other) const {
  assert(m_cols == other.rows());

  Matrix res = Matrix(m_rows, other.cols());

  for (size_t r = 0; r < m_rows; r++) {
    for (size_t c = 0; c < other.cols(); c++) {
      float sum = 0;
      for (size_t i = 0; i < other.rows(); i++) {
        sum += other.get(i, c) * get(r, i);
      }
      res.set(r, c, sum);
    }
  }

  return res;
}

Matrix Matrix::elem_multiply(const Matrix &other) const {
  assert(same_shape(other));

  Matrix out = Matrix(m_rows, m_cols);
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      out.set(i, j, get(i, j) * other.get(i, j));
    }
  }

  return out;
}

Matrix &Matrix::elem_multiply_inplace(const Matrix &other) {
  assert(same_shape(other));

  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      set(i, j, get(i, j) * other.get(i, j));
    }
  }

  return *this;
}

Matrix Matrix::add(float x) const {
  Matrix out = Matrix(m_rows, m_cols);
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      out.set(i, j, get(i, j) + x);
    }
  }
  return out;
}

Matrix &Matrix::add_inplace(float x) {
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      m_elements[i * m_cols + j] += x;
    }
  }

  return *this;
}

Matrix Matrix::add(const Matrix &other) const {
  assert(other.rows() == m_rows && other.cols() == m_cols);
  Matrix res = Matrix(m_rows, m_cols);

  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      res.set(i, j, get(i, j) + other.get(i, j));
    }
  }

  return res;
}

Matrix &Matrix::add_inplace(const Matrix &other) {
  assert(other.rows() == m_rows && other.cols() == m_cols);
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      set(i, j, get(i, j) + other.get(i, j));
    }
  }
  return *this;
}

Matrix Matrix::pow_elem(float p) const {
  return apply_function([p](float x) -> float { return std::pow(x, p); });
}

Matrix &Matrix::pow_elem_inplace(float p) {
  return apply_function_inplace(
      [p](float x) -> float { return std::pow(x, p); });
}

Matrix Matrix::transpose() const {
  Matrix t = Matrix(m_cols, m_rows);

  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      t.set(j, i, get(i, j));
    }
  }

  return t;
}

Matrix Matrix::apply_function(std::function<float(float)> func) const {
  Matrix out = Matrix(m_rows, m_cols);
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      out.set(i, j, func(get(i, j)));
    }
  }
  return out;
}

Matrix &Matrix::apply_function_inplace(std::function<float(float)> func) {
  for (size_t i = 0; i < m_rows; i++) {
    for (size_t j = 0; j < m_cols; j++) {
      set(i, j, func(get(i, j)));
    }
  }
  return *this;
}

bool Matrix::same_shape(const Matrix &other) const {
  return (m_cols == other.cols() && m_rows == other.rows());
}

void Matrix::print() const {
  for (size_t i = 0; i < m_rows; i++) {
    std::cout << "| ";
    for (size_t j = 0; j < m_cols; j++) {
      std::cout << get(i, j) << " ";
    }
    std::cout << "|";
    std::cout << "\n";
  }
}

Matrix Matrix::with_same_shape(const Matrix &other) {
  return Matrix(other.rows(), other.cols());
}

} // namespace Dendrite

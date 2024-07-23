#ifndef COL_VEC_H
#define COL_VEC_H

#include "math/Matrix.hpp"
class ColumnVector : Matrix {
public:
  ColumnVector(int length) : Matrix(length, 1) {}
  ColumnVector(int length, float fillVal) : Matrix(length, 1, fillVal) {}
  template <int length>
  ColumnVector(float (&data)[length]) : Matrix(data, length, 1) {}
  ColumnVector(ColumnVector &vec) : Matrix(vec) {}

  float get(int i) { return Matrix::get(i, 0); }

  int len() const { return Matrix::rows(); }

  void set(int i, float val) { Matrix::set(i, 0, val); }
};

#endif // !COL_VEC_H

#include "math/Matrix.hpp"

int main() {
  // clang-format off
  float m1[2][3] = {
    {1, 2, 3},
    {4, 5, 6}
  };
  float m2[3][2] = {
    {10, 11},
    {20, 21},
    {30, 31}
  };
  // clang-format on

  Matrix mat = Matrix(m1);
  Matrix mat2 = Matrix(m2);
  Matrix product = mat.multiply(mat2);

  product.print();
}

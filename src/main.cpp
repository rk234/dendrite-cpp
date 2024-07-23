#include "math/matrix.hpp"
#include <iostream>

int main() {
  float m1[2][3] = {{1, 2, 3}, {4, 5, 6}};
  float m2[3][2] = {{10, 11}, {20, 21}, {30, 31}};
  Matrix mat = Matrix(m1);
  Matrix mat2 = Matrix(m2);
  Matrix product = mat.multiply(mat2);

  for (int i = 0; i < product.rows(); i++) {
    for (int j = 0; j < product.rows(); j++) {
      std::cout << product.get(i, j) << " ";
    }
    std::cout << "\n";
  }
}

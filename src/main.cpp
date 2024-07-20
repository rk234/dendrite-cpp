#include "math/matrix.hpp"
#include <iostream>

int main() {
  Matrix mat = Matrix(3, 3);
  mat.set(0, 0, 1.0f);
  Matrix copy = Matrix(mat);
  std::cout << mat.get(0, 0) << "\n";
}

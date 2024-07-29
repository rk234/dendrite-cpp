#ifndef MNIST_H
#define MNIST_H
#include "math/Matrix.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
class Mnist {
private:
  std::optional<Matrix> trainImages;
  std::optional<Matrix> trainLabels;
  std::optional<Matrix> testImages;
  std::optional<Matrix> testLabels;

  Matrix loadImages(std::filesystem::path file) {
    std::ifstream stream(file, std::ios_base::binary);
    uint32_t magicNum;
    stream.read(&magicNum, 4);
  }

public:
  void load(std::filesystem::path directory) {}
};
#endif // !MNIST_H

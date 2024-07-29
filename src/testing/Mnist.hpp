#ifndef MNIST_H
#define MNIST_H
#include "math/Matrix.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
class Mnist {
private:
  std::optional<Matrix> trainImages;
  std::optional<Matrix> trainLabels;
  std::optional<Matrix> testImages;
  std::optional<Matrix> testLabels;

  int read_i32(std::ifstream &stream) {
    int val;
    stream.read(reinterpret_cast<char *>(&val), 4);
    return reverse_int(val);
  }

  int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  }

  Matrix load_labels(std::filesystem::path file) {
    std::ifstream stream(file);

    if (!stream.is_open()) {
      std::cerr << "FILE " << file << " NOT FOUND!";
      return Matrix(1, 1);
    }

    int magicNum = read_i32(stream);
    std::cout << "Magic num: " << magicNum << "\n";

    int numLabels = read_i32(stream);
    std::cout << "Number of labels: " << numLabels << "\n";

    Matrix oneHotEncodedOutputs = Matrix(10, numLabels, 0);

    for (int i = 0; i < numLabels; i++) {
      int label;
      stream.read(reinterpret_cast<char *>(&label), 1);
      std::cout << "label: " << label << "\n";
      oneHotEncodedOutputs.set(label, i, 1.0f);
    }

    return oneHotEncodedOutputs;
  }

  Matrix load_images(std::filesystem::path file) {
    std::ifstream stream(file);

    if (!stream.is_open()) {
      std::cerr << "FILE " << file << " NOT FOUND!";
      return Matrix(1, 1);
    }

    int magicNum = read_i32(stream);
    std::cout << "Magic num: " << magicNum << "\n";

    int numImages = read_i32(stream);
    std::cout << "Number of images: " << numImages << "\n";

    int numRows = read_i32(stream);
    int numCols = read_i32(stream);
    std::cout << "rows: " << numRows << ", cols: " << numCols << "\n";

    Matrix images = Matrix(numRows * numCols, numImages);

    for (int i = 0; i < numImages; i++) {
      for (int c = 0; c < numCols; c++) {
        for (int r = 0; r < numRows; r++) {
          int pixel;
          stream.read(reinterpret_cast<char *>(&pixel), 1);
          // std::cout << "pixel: " << pixel << "\n";

          images.set(c * numRows + r, i, (float)pixel / 255.0f);
        }
      }
    }

    return images;
  }

public:
  void load(std::filesystem::path directory) {
    trainImages = load_images(directory / "train-images.idx3-ubyte");
    trainLabels = load_labels(directory / "train-labels.idx1-ubyte");
    testImages = load_images(directory / "t10k-images.idx3-ubyte");
    testLabels = load_labels(directory / "t10k-labels.idx1-ubyte");
    // std::cout << "OUT: ";
    //
    // for (int i = 0; i < 10; i++) {
    //   std::cout << trainLabels->get(i, 0) << " ";
    // }
  }

  void display_image(const Matrix &images, int size, int imageIdx) {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        if (images.get(i * size + j, imageIdx) > 0) {
          std::cout << "██";
        } else {
          std::cout << "  ";
        }
      }
      std::cout << "\n";
    }
  }
};
#endif // !MNIST_H

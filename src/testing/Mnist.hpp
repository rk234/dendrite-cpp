#ifndef MNIST_H
#define MNIST_H
#include "math/Matrix.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
class Mnist {
private:
  std::optional<Dendrite::Matrix> trainImages;
  std::optional<Dendrite::Matrix> trainLabels;
  std::optional<Dendrite::Matrix> testImages;
  std::optional<Dendrite::Matrix> testLabels;

  int read_i32(std::ifstream &stream) {
    int val = 0;
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

  Dendrite::Matrix load_labels(std::filesystem::path file) {
    std::cout << "Loading MNIST labels at " << file << "...";
    std::ifstream stream(file);

    if (!stream.is_open()) {
      std::cerr << "FILE " << file << " NOT FOUND!";
      return Dendrite::Matrix(1, 1);
    }

    read_i32(stream);
    // std::cout << "Magic num: " << magicNum << "\n";

    int numLabels = read_i32(stream);
    // std::cout << "Number of labels: " << numLabels << "\n";

    Dendrite::Matrix oneHotEncodedOutputs = Dendrite::Matrix(10, numLabels, 0);

    for (int i = 0; i < numLabels; i++) {
      int label = 0;
      stream.read(reinterpret_cast<char *>(&label), 1);
      // std::cout << "label: " << label << "\n";
      oneHotEncodedOutputs.set(label, i, 1.0f);
    }

    stream.close();
    std::cout << "Done!\n";

    return oneHotEncodedOutputs;
  }

  Dendrite::Matrix load_images(std::filesystem::path file) {
    std::cout << "Loading MNIST images at " << file << "...";
    std::ifstream stream(file);

    if (!stream.is_open()) {
      std::cerr << "FILE " << file << " NOT FOUND!";
      return Dendrite::Matrix(1, 1);
    }

    int magicNum = read_i32(stream);
    // std::cout << "Magic num: " << magicNum << "\n";

    int numImages = read_i32(stream);
    // std::cout << "Number of images: " << numImages << "\n";

    int numRows = read_i32(stream);
    int numCols = read_i32(stream);
    // std::cout << "rows: " << numRows << ", cols: " << numCols << "\n";

    Dendrite::Matrix images = Dendrite::Matrix(numRows * numCols, numImages);

    for (int i = 0; i < numImages; i++) {
      for (int c = 0; c < numCols; c++) {
        for (int r = 0; r < numRows; r++) {
          int pixel = 0;
          stream.read(reinterpret_cast<char *>(&pixel), 1);
          // std::cout << "pixel: " << pixel << "\n";

          images.set(c * numRows + r, i, (float)pixel / 255.0f);
        }
      }
    }

    stream.close();
    std::cout << "Done!\n";
    return images;
  }

public:
  void load(std::filesystem::path directory) {
    trainImages = load_images(directory / "train-images.idx3-ubyte");
    trainLabels = load_labels(directory / "train-labels.idx1-ubyte");
    testImages = load_images(directory / "t10k-images.idx3-ubyte");
    testLabels = load_labels(directory / "t10k-labels.idx1-ubyte");
  }

  const std::optional<Dendrite::Matrix> &
  get_train_images() const { // Each column is one image
    return trainImages;
  }
  const std::optional<Dendrite::Matrix> &
  get_train_labels() const { // Each column is a one-hot encoded output
    return trainLabels;
  }
  const std::optional<Dendrite::Matrix> &
  get_test_images() const { // Each column is one image
    return testImages;
  }
  const std::optional<Dendrite::Matrix> &
  get_test_labels() const { // Each column is a one-hot encoded output
    return testLabels;
  }

  void display_image(const Dendrite::Matrix &images, int size,
                     int imageIdx) const {
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

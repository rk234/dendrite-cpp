#ifndef ACTIVATION_H
#define ACTIVATION_H
class ActivationFunction {
public:
  virtual float activate(float input) = 0;
};
#endif // !ACTIVATION_H

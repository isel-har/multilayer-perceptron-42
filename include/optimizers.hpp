#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "layer.hpp"

#include <vector>


class Optimizer
{ // pure base class
  public:
    virtual void update(std::vector<Layer>&) {}
    virtual ~Optimizer() {}
};

class GradientDescent : public Optimizer
{
  private:
    double learning_rate;

  public:
    GradientDescent(double);
    void update(std::vector<Layer>&) override;
};

// class Adam:public Optimizer {
// public:
//     // Adam();
//     Adam(double);
//     void update(std::vector<Layer>&) override;
// };

#endif
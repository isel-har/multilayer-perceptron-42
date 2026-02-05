#ifndef OPTIMIZERS_HPP
#define OPTIMIZERS_HPP

#include "layer.hpp"

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

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

class Adam : public Optimizer
{
  private:
    double learning_rate;
    int t;
    std::vector<MatrixXd>     weights_momentums_vec;
    std::vector<MatrixXd>     weights_rms_props_vec;
    std::vector<RowVectorXd>  biases_momentums_vec;
    std::vector<RowVectorXd>  biases_rms_props_vec;
  
  public:
      Adam(double, const std::vector<Layer>&);
      void update(std::vector<Layer>&) override;
};

#endif
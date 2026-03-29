#ifndef METRICS_HPP
#define METRICS_HPP

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class Metric
{
  public:
    virtual double compute(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const = 0;
    virtual ~Metric() {};
};

class Accuracy : public Metric
{
  public:
    double compute(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const override;
};

class Precision : public Metric
{
  public:
    double compute(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const override;
};

#endif

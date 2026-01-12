#ifndef METRICS_HPP
#define METRICS_HPP

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

// difference between metrics, when use argmax and why, when use actual probs!!!

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

class BinarycrossEntropy : public Metric
{
  public:
    double compute(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const override;
    // double compute(const Eigen::MatrixXd&, const Eigen::MatrixXd&) const override;
};

#endif

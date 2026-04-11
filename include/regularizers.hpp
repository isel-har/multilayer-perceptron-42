#pragma once

#include <Eigen/Dense>


class L2Regularizer {
public:
    double l2;
    L2Regularizer(double);
    double operator()(const Eigen::MatrixXd &weights);
    Eigen::MatrixXd gradient(const Eigen::MatrixXd &weights);
};
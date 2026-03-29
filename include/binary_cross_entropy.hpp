#pragma once

#include <Eigen/Dense>

class BinaryCrossEntropy {

private:
    double epsilon = 1e-15;
public:
    BinaryCrossEntropy(double);

    double forward(const Eigen::MatrixXd& probs, const Eigen::MatrixXd& ybatch);
    Eigen::MatrixXd backward(const Eigen::MatrixXd& probs, const Eigen::MatrixXd& ybatch);
};
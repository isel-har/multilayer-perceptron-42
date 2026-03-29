#include "binary_cross_entropy.hpp"

BinaryCrossEntropy::BinaryCrossEntropy(double epsilon):epsilon(epsilon){}

double BinaryCrossEntropy::forward(const Eigen::MatrixXd& probs, const Eigen::MatrixXd& ybatch) {
    auto p = probs.array().max(epsilon).min(1.0 - epsilon);
    auto y = ybatch.array();

    return -(y * p.log() + (1.0 - y) * (1.0 - p).log()).mean();
}

Eigen::MatrixXd BinaryCrossEntropy::backward(const Eigen::MatrixXd& probs, const Eigen::MatrixXd& ybatch) {
    auto p = probs.array().max(epsilon).min(1.0 - epsilon);
    auto y = ybatch.array();

    return ((p - y) / (p * (1.0 - p)) / y.size()).matrix();
}
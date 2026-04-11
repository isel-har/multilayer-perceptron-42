#include "regularizers.hpp"

L2Regularizer::L2Regularizer(double l2_): l2(l2_) {}

double L2Regularizer::operator()(const Eigen::MatrixXd &weights) {
    return l2 * weights.squaredNorm();
}

Eigen::MatrixXd L2Regularizer::gradient(const Eigen::MatrixXd &weights) {
        return 2.0 * l2 * weights;
}
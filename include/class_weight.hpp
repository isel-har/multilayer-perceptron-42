#pragma once

#include <utility>
#include <Eigen/Dense>

std::pair<double, double> compute_binary_class_weight(const Eigen::MatrixXd &y);
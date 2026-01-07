#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using namespace Eigen;

MatrixXd    relu(const MatrixXd&, bool);
MatrixXd    sigmoid(const MatrixXd&, bool);
MatrixXd    softmax(const MatrixXd&, bool);

#endif


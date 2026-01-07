#ifndef CSV_TO_EIGEN_HPP
#define CSV_TO_EIGEN_HPP

#include "rapidcsv.h"
#include <Eigen/Dense>

using namespace Eigen;

struct xy_eigen {
    MatrixXd Y;
    MatrixXd X;
};

xy_eigen    csv_to_eigen(const std::string &);

#endif
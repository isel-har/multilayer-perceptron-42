#ifndef CSV_TO_EIGEN_HPP
#define CSV_TO_EIGEN_HPP

#include "rapidcsv.h"
#include "scaler.hpp"

#include <Eigen/Dense>
#include <utility>

using namespace Eigen;

struct DatasetSplit
{
    MatrixXd X_train;
    MatrixXd y_train;
    MatrixXd X_val;
    MatrixXd y_val;
};

MatrixXd        doc_to_eigen_encoded(const rapidcsv::Document& doc);
MatrixXd        doc_to_eigen(const rapidcsv::Document& doc);
DatasetSplit    train_val_split();



#endif
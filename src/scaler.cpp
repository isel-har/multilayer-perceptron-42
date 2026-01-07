#include "scaler.hpp"

Eigen::MatrixXd StandardScaler(const Eigen::MatrixXd &X) {

    Eigen::RowVectorXd  mean = X.colwise().mean();
    Eigen::MatrixXd     centered = X.rowwise() - mean;
    Eigen::RowVectorXd  std_dev = (centered.array().square().colwise().sum() / X.rows()).sqrt();

    for (int i = 0; i < std_dev.size(); ++i) {
        if (std_dev(i) == 0) std_dev(i) = 1.0;
    }
    Eigen::MatrixXd X_scaled = centered.array().rowwise() / std_dev.array();
    return X_scaled;
}
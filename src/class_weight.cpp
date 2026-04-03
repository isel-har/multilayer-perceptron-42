#include "class_weight.hpp"


std::pair<double, double> compute_binary_class_weight(const Eigen::MatrixXd &y)
{
    std::pair<double, double> class_weight;
    double N = static_cast<double>(y.rows());

    double count_1 = static_cast<double>((y.col(0).array() == 1.0).count());
    double count_0 = static_cast<double>((y.col(0).array() == 0.0).count());

    class_weight.first = N / (2 * count_1);
    class_weight.second = N / (2 * count_0);
    return class_weight;
}

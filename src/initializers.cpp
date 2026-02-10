#include "initializers.hpp"


MatrixXd Random::init(unsigned int rows, unsigned int cols) {
    return MatrixXd::Random(rows, cols);
}

MatrixXd He::init(unsigned int rows, unsigned int cols) {

    double std_dev = std::sqrt(2.0 / rows);
    return  MatrixXd::Random(rows, cols) * std_dev;
}

MatrixXd Xavier::init(unsigned int rows, unsigned int cols)
{
    double std_dev = std::sqrt(2.0 / (rows + cols));
    return MatrixXd::Random(rows, cols) * std_dev;
}

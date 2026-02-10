#include "initializers.hpp"

MatrixXd Initializer::random_init(unsigned int rows, unsigned int cols)
{
    MatrixXd m =  MatrixXd(rows, cols);
    m = MatrixXd::Random(rows, cols);
    return m;
}

MatrixXd Initializer::he_init(unsigned int rows, unsigned int cols)
{
    double std_dev = std::sqrt(2.0 / rows);
    MatrixXd m = MatrixXd(rows, cols);
    m = MatrixXd::Random(rows, cols) * std_dev;
    return  m;
}


MatrixXd Initializer::xavier_init(unsigned int rows, unsigned int cols)
{
    double std_dev = std::sqrt(2.0 / (rows + cols));
    MatrixXd m =  MatrixXd(rows, cols);
    m = MatrixXd::Random(rows, cols) * std_dev;
    return  m;
}
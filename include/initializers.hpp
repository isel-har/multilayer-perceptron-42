#ifndef INITIALIZERS_HPP
#define INITIALIZERS_HPP

#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

class Initializer {
public:
    static MatrixXd    random_init(unsigned int rows, unsigned int cols);
    static MatrixXd    he_init(unsigned int rows, unsigned int cols);
    static MatrixXd    xavier_init(unsigned int rows, unsigned int cols);
};

#endif

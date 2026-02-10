#ifndef INITIALIZERS_HPP
#define INITIALIZERS_HPP

#include <Eigen/Dense>

using namespace Eigen;

class Random {
public:
    static MatrixXd    init(unsigned int rows, unsigned int cols);
};


class He {
public:
    static MatrixXd    init(unsigned int rows, unsigned int cols);
};


class Xavier {
public:
    static MatrixXd    init(unsigned int rows, unsigned int cols);
};

#endif

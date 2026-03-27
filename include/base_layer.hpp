#pragma once

#include <Eigen/Dense>

using namespace Eigen;


class BaseLayer
{
public:
    unsigned int input_size;
    unsigned int output_size;

    virtual MatrixXd backward(const MatrixXd &) = 0;
    virtual MatrixXd forward(const MatrixXd &)  = 0;

    virtual ~BaseLayer() = 0;
};

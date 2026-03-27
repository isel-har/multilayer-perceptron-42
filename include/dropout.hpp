#pragma once

#include "base_layer.hpp"
// #include <Eigen/Dense>

// using namespace Eigen;

class Dropout : public BaseLayer
{
public:

    float           rate;
    float           keep_prob;
    bool            training_phase;
    MatrixXd        mask;

    Dropout(float rate);

    MatrixXd    forward(const MatrixXd &);
    MatrixXd    backward(const MatrixXd &);
};


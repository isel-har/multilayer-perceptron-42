#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <functional>
#include "activations.hpp"

using namespace Eigen;

class Layer
{
  private:
    static std::unordered_map<std::string, std::function<MatrixXd(const MatrixXd&, bool)>>
        activationMap;

  public:
    unsigned int                                   size;
    unsigned int                                   input_shape;
    std::function<MatrixXd(const MatrixXd&, bool)> activation;

    std::string activation__;

    MatrixXd    weights;
    RowVectorXd biases;

    MatrixXd    weights_gradients;
    RowVectorXd biases_gradients;

    /*
        layer caches
    */
    MatrixXd output_cache;
    MatrixXd input_cache; // to change!
    MatrixXd z_cache;

    Layer(unsigned int, unsigned int, const std::string&, bool);
    Layer();
    ~Layer();

    MatrixXd forward(const MatrixXd&);
    MatrixXd backward(const MatrixXd&);
};
static std::unordered_map<std::string, std::function<MatrixXd(const MatrixXd&, bool)>>
    activationMap;

#endif
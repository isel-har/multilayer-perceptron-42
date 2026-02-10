#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <functional>
#include <cassert>

#include "activations.hpp"
#include "initializers.hpp"
#include "json.hpp"


using namespace Eigen;
using json = nlohmann::json;

class Layer
{
  private:
    static std::unordered_map<std::string, std::function<MatrixXd(const MatrixXd&, bool)>>
        activationMap;
    static std::unordered_map<std::string, std::function<MatrixXd(unsigned int rows, unsigned int cols)>>
        initializersMap;

  public:
    unsigned int                                   size;
    unsigned int                                   input_shape;
    std::function<MatrixXd(const MatrixXd&, bool)> activation;

    std::string activation__;

    MatrixXd    weights;
    RowVectorXd biases;

    MatrixXd    weights_gradients;
    RowVectorXd biases_gradients;

    // Initializer *initializer_ptr;
    /*
        layer caches
    */
    MatrixXd output_cache;
    MatrixXd input_cache; // to change!
    MatrixXd z_cache;


    // Layer(Initializer &initializer, const std::string& activation);
    Layer(unsigned int, unsigned int, const std::string&, bool);
    Layer(const json &hidden_layer, unsigned int shape);
    Layer();
    ~Layer();

    MatrixXd forward(const MatrixXd&);
    MatrixXd backward(const MatrixXd&);
};
static std::unordered_map<std::string, std::function<MatrixXd(const MatrixXd&, bool)>>
    activationMap;

static std::unordered_map<std::string, std::function<MatrixXd(unsigned int rows, unsigned int cols)>>
        initializersMap;


#endif
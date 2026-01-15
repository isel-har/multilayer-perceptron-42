#include "layer.hpp"

#include "activations.hpp"

std::unordered_map<std::string, std::function<MatrixXd(const MatrixXd&, bool)>>
    Layer::activationMap = {{"relu", relu}, {"sigmoid", sigmoid}, {"softmax", softmax}};


Layer::Layer():size(0), input_shape(0)
{
    this->activation = this->activationMap["relu"];

    this->activation__      = "relu";
    this->weights  = MatrixXd::Zero(input_shape, size);// * std_dev;
    this->biases = RowVectorXd::Zero(size);
    this->weights_gradients = MatrixXd::Zero(input_shape, size);
    this->biases_gradients  = RowVectorXd::Zero(size);
}

Layer::Layer(unsigned int input_shape, unsigned int size, const std::string& activation_)
    : size(size), input_shape(input_shape)
{
    this->activation = this->activationMap[activation_];

    this->activation__      = activation_;
    double std_dev = std::sqrt(2.0 / input_shape);
    this->weights  = MatrixXd::Random(input_shape, size) * std_dev;
    this->biases = RowVectorXd::Zero(size);
    this->weights_gradients = MatrixXd::Zero(input_shape, size);
    this->biases_gradients  = RowVectorXd::Zero(size);
}

MatrixXd Layer::forward(const MatrixXd& input)
{
    this->input_cache  = input;
    this->z_cache      = (input * weights).rowwise() + biases;
    this->output_cache = this->activation(this->z_cache, false);
    return this->output_cache;
}

MatrixXd Layer::backward(const MatrixXd& d_out)
{
    MatrixXd d_Z            = d_out.cwiseProduct(this->activation(this->z_cache, true));
    this->weights_gradients = this->input_cache.transpose() * d_Z;
    this->biases_gradients  = d_Z.colwise().sum();
    MatrixXd d_X            = d_Z * this->weights.transpose();
    return d_X;
}

Layer::~Layer() {}

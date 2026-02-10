#include "layer.hpp"


std::unordered_map<std::string, std::function<MatrixXd(const MatrixXd&, bool)>>
    Layer::activationMap = {{"relu", relu}, {"sigmoid", sigmoid}, {"softmax", softmax}};

std::unordered_map<std::string, std::function<MatrixXd(unsigned int rows, unsigned int cols)>>
    Layer::initializersMap = {{"he", Initializer::he_init}, {"xavier", Initializer::xavier_init}, {"random", Initializer::random_init}};

Layer::Layer():size(0), input_shape(0)
{
    this->activation = this->activationMap["relu"];

    this->activation__      = "relu";
    this->weights  = MatrixXd::Zero(input_shape, size);
    this->biases = RowVectorXd::Zero(size);
    this->weights_gradients = MatrixXd::Zero(input_shape, size);
    this->biases_gradients  = RowVectorXd::Zero(size);
}

Layer::Layer(const json &hidden_layer_json,  unsigned int input_shape_)
{
    std::string initializer_ = hidden_layer_json.value("initializer", "he");
    std::string activation_  = hidden_layer_json.value("activation", "relu");
    this->activation__ = activation_;

    unsigned int size_ = hidden_layer_json["size"];
    this->activation  = this->activationMap[activation_];

    size = size_;
    input_shape = input_shape_;
    std::function<MatrixXd(unsigned int rows, unsigned int cols)> initializer = this->initializersMap[initializer_];
    
    if (!initializer)
        throw std::runtime_error("invalid initializer, (xavier, he, random)\n");

    if (!this->activation)
        throw std::runtime_error("invalid activation function, (sigmoid, relu)\n");
    
    this->weights = initializer(input_shape, size);
    
    this->biases = RowVectorXd::Zero(size);
    this->weights_gradients = MatrixXd::Zero(input_shape, size);
    this->biases_gradients  = RowVectorXd::Zero(size);
}

Layer::Layer(unsigned int input_shape,
             unsigned int size,
             const std::string& activation_,
             bool zeros)
    : size(size), input_shape(input_shape)
{
    this->activation = this->activationMap[activation_];
    if (!this->activation)
        std::cout << "activation not found\n";
    this->activation__ = activation_;

    if (zeros)
    {
        this->weights = MatrixXd::Zero(input_shape, size);
    }
    else
    {
        double std_dev = std::sqrt(2.0 / input_shape);
        this->weights = MatrixXd::Random(input_shape, size) * std_dev;
    }

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

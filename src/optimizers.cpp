#include "optimizers.hpp"

GradientDescent::GradientDescent(double learning_rate) : learning_rate(learning_rate) {}

void GradientDescent::update(std::vector<Layer>& layers)
{
    for (auto& layer : layers)
    {
        layer.weights = layer.weights - (this->learning_rate * layer.weights_gradients);
        layer.biases  = layer.biases - (this->learning_rate * layer.biases_gradients);
    }
}

Adam::Adam(double lr, const std::vector<Layer>& layers):learning_rate(lr), t(0),
    weights_momentums_vec(layers.size()),
    weights_rms_props_vec(layers.size()),
    biases_momentums_vec(layers.size()),
    biases_rms_props_vec(layers.size())
{
    for (size_t i = 0; i < layers.size(); ++i)
    {
        size_t size = layers[i].size;
        size_t input_shape = layers[i].input_shape;

        weights_momentums_vec[i] = MatrixXd::Zero(input_shape, size);
        weights_rms_props_vec[i] = MatrixXd::Zero(input_shape, size);
        biases_momentums_vec[i]  = RowVectorXd::Zero(size);
        biases_rms_props_vec[i]  = RowVectorXd::Zero(size);
    }
}
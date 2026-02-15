#include "optimizers.hpp"

GradientDescent::GradientDescent(double learning_rate) : learning_rate(learning_rate) {}

Adam::Adam(double lr, const std::vector<Layer>& layers):learning_rate(lr), t(0),
weights_momentums_vec(layers.size()),
weights_rms_props_vec(layers.size()),
biases_momentums_vec(layers.size()),
biases_rms_props_vec(layers.size())
{
    std::cout << "adam optimizer used\n";
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

void GradientDescent::update(std::vector<Layer>& layers)
{
    for (auto& layer : layers)
    {
        layer.weights = layer.weights - (this->learning_rate * layer.weights_gradients);
        layer.biases  = layer.biases  - (this->learning_rate * layer.biases_gradients);
    }
}

// m = β1 * m + (1 - β1) * gradient      ← momentum
// v = β2 * v + (1 - β2) * gradient²     ← RMSprop
// weight = weight - lr * m / sqrt(v)

void Adam::update(std::vector<Layer>& layers)
{
    const double m_dec = 0.9;
    const double rms_dec = 0.999;
    const double epsilon = 0.00000001;
    
    ++t;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        weights_momentums_vec[i] = (weights_momentums_vec[i] * m_dec) + (layers[i].weights_gradients * (1 - m_dec));
        weights_rms_props_vec[i] = (weights_rms_props_vec[i] * rms_dec) + (layers[i].weights_gradients.array().pow(2).matrix() * (1 - rms_dec));

        biases_momentums_vec[i] = (biases_momentums_vec[i] * m_dec) + (layers[i].biases_gradients * (1 - m_dec));
        biases_rms_props_vec[i] = (biases_rms_props_vec[i] * rms_dec) + (layers[i].biases_gradients.array().pow(2).matrix() * (1 - rms_dec));

        MatrixXd m_w_hat  = weights_momentums_vec[i] / (1 - std::pow(m_dec, t));
        MatrixXd v_w_hat  = weights_rms_props_vec[i] / (1 - std::pow(rms_dec, t));
  
        MatrixXd m_b_hat  = biases_momentums_vec[i] / (1 - std::pow(m_dec, t));
        MatrixXd v_b_hat  = biases_rms_props_vec[i] / (1 - std::pow(rms_dec, t));

        // INCORRECT: MatrixXd denom_w = v_w_hat.array().pow(2) + epsilon; 
        // MatrixXd denom_w = v_w_hat.array().pow(2) + epsilon;                                                                                                                                                                                                                             
        // MatrixXd denom_b = v_b_hat.array().pow(2) + epsilon;

        // layers[i].weights = layers[i].weights - (learning_rate * (m_w_hat.array() / denom_w.array().sqrt()).matrix());
        // layers[i].biases = layers[i].biases - (learning_rate * (m_b_hat.array() / denom_b.array().sqrt()).matrix());

        // CORRECT:
        MatrixXd denom_w = v_w_hat.array().sqrt() + epsilon;
        MatrixXd denom_b = v_b_hat.array().sqrt() + epsilon;

        layers[i].weights = layers[i].weights - (learning_rate * (m_w_hat.array() / denom_w.array()).matrix());
        layers[i].biases = layers[i].biases - (learning_rate * (m_b_hat.array() / denom_b.array()).matrix());

    }
}

Adam::~Adam() {

}

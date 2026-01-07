#include "optimizers.hpp"

GradientDescent::GradientDescent(double learning_rate):learning_rate(learning_rate) {
}

void    GradientDescent::update(std::vector<Layer>& layers) {

    for (auto &layer:layers) {

        layer.weights = layer.weights - (this->learning_rate * layer.weights_gradients);  
        layer.biases  = layer.biases  - (this->learning_rate * layer.biases_gradients);  
    }
}
// void    Adam::update(void) {

// }

// Adam::Adam(double learning_rate){
//     (void)learning_rate;
// }
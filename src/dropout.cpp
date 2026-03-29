// #include "dropout.hpp"
// #include <random>

// Dropout::Dropout(float rate) : rate(rate), keep_prob(1.0f - rate) {
// }

// MatrixXd Dropout::forward(const MatrixXd &input) {
//     if (!this->training) {
//         return input; // During inference, we don't drop anything
//     }

//     // Generate a random mask of 0s and 1s
//     // We use Bernoulli distribution to decide which neurons stay
//     std::default_random_engine generator;
//     std::bernoulli_distribution distribution(keep_prob);

//     // Initialize mask with same shape as input
//     mask = MatrixXd::Zero(input.rows(), input.cols());
    
//     for (int i = 0; i < input.rows(); ++i) {
//         for (int j = 0; j < input.cols(); ++j) {
//             if (distribution(generator)) {
//                 mask(i, j) = 1.0 / keep_prob; // Inverted Dropout scaling
//             }
//         }
//     }

//     return input.cwiseProduct(mask);
// }

// MatrixXd Dropout::backward(const MatrixXd &output_grad) {
//     // Gradient only flows through the neurons that weren't dropped
//     return output_grad.cwiseProduct(mask);
// }
#ifndef CHECKER_HPP
#define CHECKER_HPP

#include <stdexcept>
#include <string>
#include <fstream>

template <typename T>
T   checked_range(const T& value, const T& min, const T& max, const std::string& name)
{
    if (value < min || value > max)
        throw std::out_of_range(
            name + " must be in [" + std::to_string(min) + ", " + std::to_string(max) + "]"
        );
    return value;
}

template <typename T>
void    checked_layers(const T& layers) {

    for (size_t i = 0; i < layers.size() - 1; ++i) {
        checked_range((unsigned int)layers[i]["size"], (unsigned int)1, (unsigned int)256, "hidden_size");
    }
}

#endif

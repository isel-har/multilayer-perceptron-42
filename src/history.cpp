#include "history.hpp"

History::History(size_t size):loss(size), accuracy(size), precision(size) {
    this->vecMap["loss"]      = &this->loss;
    this->vecMap["accuracy"]  = &this->accuracy;
    this->vecMap["precision"] = &this->precision;
}

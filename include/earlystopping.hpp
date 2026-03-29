#pragma once

#include <limits>
#include <iostream>
#include "layer.hpp"


class MLPClassifier;

class EarlyStopping
{
  public:
    bool          _enabled;
    unsigned char _patience;
    unsigned char times;
    double        optimal_loss;
    bool          enabled;
    bool          restore_best_weights;
    double        _min_delta;
    
    std::vector<Layer>  best_weights;

    EarlyStopping(char patience, bool enabled, bool restore_best_weights);
    bool operator()(double val_loss, MLPClassifier* model_ptr);
};


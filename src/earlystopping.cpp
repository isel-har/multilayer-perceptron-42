#include "earlystopping.hpp"
#include "mlpclassifier.hpp"
// EarlyStopping::EarlyStopping(bool enabled)
//     : _enabled(enabled), _patience(6), optimal_loss(std::numeric_limits<double>::max())
// {
// }
EarlyStopping::EarlyStopping(char patience, bool enabled, bool restore_best_weights)
    : _enabled(enabled),
    _patience(patience),
    optimal_loss(std::numeric_limits<double>::max()),
    restore_best_weights(restore_best_weights),
    _min_delta(0.001)
{
}
bool EarlyStopping::operator()(double val_loss, MLPClassifier *model_ptr)
{
    if (!_enabled) return false;

    if (val_loss < (optimal_loss - _min_delta)) 
    {
        optimal_loss = val_loss;
        times = 0;

        if (restore_best_weights) {
            best_weights = model_ptr->get_weights(); 
        }
    }
    else 
    {
        ++times;
    }

    if (times >= _patience) 
    {
        if (restore_best_weights && !best_weights.empty()) {
            model_ptr->set_weights(std::move(best_weights));
        }
        return true; // Stop training
    }

    return false;
}
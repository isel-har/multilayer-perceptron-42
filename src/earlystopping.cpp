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
    restore_best_weights(restore_best_weights)
{
}

bool EarlyStopping::operator()(double val_loss, MLPClassifier *model_ptr)
{
    // if (model_ptr == nullptr)
    //     throw std::runtime_error("model object cannot be a null pointer");

    if (!_enabled)
        return false;
    
    if (val_loss < optimal_loss)
    {
        if (this->restore_best_weights == true)
        {
            this->best_weights = model_ptr->get_weights();
        }
        this->optimal_loss = val_loss;
        this->times        = 0;
    }
    else
        ++this->times;
    if (this->times >= this->_patience)
    {
        std::cout << "training loop stopped by early-stopping\n";
        if (this->restore_best_weights == true)
            model_ptr->set_weights(this->best_weights);

        return true;
    }
    return false;
}
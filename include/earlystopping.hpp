#ifndef EARLYSTOPPING_HPP
#define EARLYSTOPPING_HPP

#include <limits>

class EarlyStopping
{
  public:
    bool          _enabled;
    unsigned char _patience;
    unsigned char times;
    double        optimal_loss;
    bool          enabled;

    EarlyStopping(bool enabled);
    EarlyStopping(char patience, bool enabled);
    bool operator()(double loss);
};

#endif

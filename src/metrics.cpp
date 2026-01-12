#include "metrics.hpp"

double Accuracy::compute(const Eigen::MatrixXd& ypred, const Eigen::MatrixXd& ytrue) const
{
    int samples = ypred.rows();
    int correct = 0;

    for (int i = 0; i < samples; ++i)
    {
        if (ypred(i, 0) == ytrue(i, 0))
            ++correct;
    }
    return static_cast<double>(correct) / samples;
}

double Precision::compute(const Eigen::MatrixXd& ypred, const Eigen::MatrixXd& ytrue) const
{
    int    samples        = ypred.rows();
    double true_postives  = 0;
    double false_postives = 0;

    for (int i = 0; i < samples; ++i)
    {
        if (ypred(i, 0) == ytrue(i, 0) && ytrue(i, 0) == 0)
            ++true_postives;
        else if (ypred(i, 0) == ytrue(i, 0))
            ++false_postives;
    }
    return true_postives / (true_postives + false_postives);
}

double BinarycrossEntropy::compute(const Eigen::MatrixXd& ypred, const Eigen::MatrixXd& ytrue) const
{
    constexpr double   eps     = 1e-12;
    const unsigned int samples = ytrue.rows();

    Eigen::ArrayXXd p = ypred.array().min(1.0 - eps).max(eps);
    Eigen::ArrayXXd y = ytrue.array();

    double loss = -(y * p.log() + (1.0 - y) * (1.0 - p).log()).sum() / samples;

    return loss;
}

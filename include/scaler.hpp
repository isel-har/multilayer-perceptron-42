#ifndef SCALER_HPP
#define SCALER_HPP

#include <Eigen/Dense>

using namespace Eigen;

class Scaler
{
    public:
        MatrixXd    mean;
        MatrixXd    std;

        
        MatrixXd    fit_transform(const Eigen::MatrixXd&);
        void        save() const;
};

#endif

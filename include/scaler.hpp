#ifndef SCALER_HPP
#define SCALER_HPP

#include <Eigen/Dense>
#include <fstream>

using namespace Eigen;

class Scaler
{
    public:
        RowVectorXd mean;
        RowVectorXd std_dev;
        bool        loaded;
        
        Scaler(const std::string &);
        Scaler();
        void   fit_transform(MatrixXd&);
};

#endif

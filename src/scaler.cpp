#include "scaler.hpp"

Scaler::Scaler():mean(0), std_dev(0), loaded(false)
{}

Scaler::Scaler(const std::string& params_path)
{
    auto params_file = std::ifstream(params_path);
    if (!params_file.is_open())
        throw std::runtime_error("failed to open " + params_path);
    
    int size;
    params_file.read(reinterpret_cast<char*>(&size), sizeof(int));
    mean = RowVectorXd(1, size);
    std_dev  = RowVectorXd(1, size);
    params_file.read(reinterpret_cast<char*>(mean.data()), size * sizeof(double));
    params_file.read(reinterpret_cast<char*>(std_dev.data()), size * sizeof(double));
    params_file.close();
    loaded = true;
}

void    Scaler::fit_transform(MatrixXd& X)
{
    if (!loaded)
        throw std::runtime_error("scaler params not loaded.");
    
    if (X.cols() != mean.size())
        throw std::runtime_error(
            "Scaler mismatch: X.cols() = " + std::to_string(X.cols()) +
            ", mean.size() = " + std::to_string(mean.size())
        );

    X.rowwise() -= this->mean; // mean, std = RowVectorXd
    X.array().rowwise() /= this->std_dev.array();
}

#include "train_val_split.hpp"


t_split train_val_split(const std::string &train_path, const std::string &val_path)
{
    t_split datasplit;
    Scaler  scaler;
    std::pair<MatrixXd, MatrixXd> train_dataset = csv_to_eigen(train_path);
    std::pair<MatrixXd, MatrixXd> val_dataset   = csv_to_eigen(val_path);

    datasplit.X_train = scaler.StandardScaler(train_dataset.first, false);
    datasplit.y_train = train_dataset.second;
    // problem
    datasplit.X_val   = scaler.StandardScaler(val_dataset.first, false);
    datasplit.y_val   = val_dataset.second;

    std::cout << "X_train shape: (" << datasplit.X_train.rows() <<", "<<datasplit.X_train.cols() << ")\n";
    std::cout << "X_val shape: (" << datasplit.X_val.rows() <<", "<<datasplit.X_val.cols() << ")\n";

    scaler.save();

    return datasplit;
}
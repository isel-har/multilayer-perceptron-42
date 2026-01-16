#include "csv_to_eigen.hpp"

MatrixXd    doc_to_eigen(const rapidcsv::Document &doc)
{
    MatrixXd    X;
    size_t      rowsize;
    size_t      colsize;

    rowsize = doc.GetRowCount();
    colsize = doc.GetColumnCount();
    X = MatrixXd(rowsize, colsize);

    for (size_t i = 0; i < colsize; ++i)
    {
        std::vector<double> col = doc.GetColumn<double>(i);
        X.col(i)                = Map<VectorXd>(col.data(), col.size());
    }
    return X;
}

MatrixXd    doc_to_eigen_encoded(const rapidcsv::Document& doc)
{
    const std::vector<char>& yv = doc.GetColumn<char>(1);

    MatrixXd Y = MatrixXd::Zero(yv.size(), 2);
    for (size_t i = 0; i < yv.size(); ++i)
    {
        size_t index        = (yv[i] == 'M') ? 0 : 1;
        Y(i, index) = 1.0;
    }
    return Y;
}

DatasetSplit    train_val_split()
{
    auto scaler = Scaler("scaler_params.bin");
    DatasetSplit datasplit;
    
    rapidcsv::Document trainDoc("data/data_train.csv", rapidcsv::LabelParams(-1, -1));
    rapidcsv::Document valDoc("data/data_val.csv", rapidcsv::LabelParams(-1, -1));

    datasplit.y_train = doc_to_eigen_encoded(trainDoc);
    datasplit.y_val   = doc_to_eigen_encoded(valDoc);

    trainDoc.RemoveColumn(1);
    valDoc.RemoveColumn(1);

    datasplit.X_train = doc_to_eigen(trainDoc);
    datasplit.X_val   = doc_to_eigen(valDoc);

    scaler.fit_transform(datasplit.X_train); // scale here!!!!
    scaler.fit_transform(datasplit.X_val);

    return datasplit;
}
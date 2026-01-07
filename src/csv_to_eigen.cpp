#include "csv_to_eigen.hpp"

xy_eigen    csv_to_eigen(const std::string &path) {

    xy_eigen xymat;
    size_t rowsize;
    size_t colsize;

    rapidcsv::Document doc(path, rapidcsv::LabelParams(-1, -1));
    rowsize = doc.GetRowCount();
    
    const std::vector<char> &yv = doc.GetColumn<char>(1);
    xymat.Y = MatrixXd::Zero(yv.size(), 2);

    for (size_t i = 0; i < yv.size(); ++i) {

        size_t index = (yv[i] == 'M') ? 0 : 1;
        xymat.Y(i, index) = 1.0;
    }
    doc.RemoveColumn(1);
    colsize = doc.GetColumnCount();
    
    xymat.X = MatrixXd(rowsize, colsize);

    for (size_t i = 0; i < colsize; ++i) {
        std::vector<double> col = doc.GetColumn<double>(i);
        xymat.X.col(i) = Map<VectorXd>(col.data(), col.size());
    }
    return xymat;
}
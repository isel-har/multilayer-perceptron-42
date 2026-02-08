#include "save_split_scaler.hpp"

rapidcsv::Document shuffle_rows(const rapidcsv::Document& doc)
{
    size_t rowCount = doc.GetRowCount();
    rapidcsv::Document shuffled_doc("", rapidcsv::LabelParams(-1, -1));

    std::vector<size_t> indices(rowCount);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < rowCount; ++i)
        shuffled_doc.SetRow(i, doc.GetRow<std::string>(indices[i]));

    return shuffled_doc;
}

void    save_split(const rapidcsv::Document& shuffled_doc, size_t val_size)
{
    size_t rowCount = shuffled_doc.GetRowCount();
    rapidcsv::Document trainDoc("", rapidcsv::LabelParams(-1, -1));
    rapidcsv::Document valDoc("", rapidcsv::LabelParams(-1, -1));

    size_t val_rows   = (rowCount * val_size) / 100;
    size_t train_rows = rowCount - val_rows;

    size_t index = 0;

    for (; index < train_rows; ++index)
        trainDoc.SetRow(index, shuffled_doc.GetRow<std::string>(index));
    for (size_t i = 0; i < val_rows; ++i)
    {
        valDoc.SetRow(i, shuffled_doc.GetRow<std::string>(index));
        ++index;
    }
    trainDoc.Save("data/data_train.csv");
    valDoc.Save("data/data_val.csv");
    std::cout << "data_train.csv and data_val.csv saved in data.\n";
}

void    save_scale(rapidcsv::Document& doc)
{
    doc.RemoveColumn(1);
    MatrixXd X = doc_to_eigen(doc);
    RowVectorXd mean     = X.colwise().mean();
    MatrixXd    centered = X.rowwise() - mean;
    RowVectorXd std_dev  = (centered.array().square().colwise().sum() / X.rows()).sqrt();

    for (int i = 0; i < std_dev.size(); ++i)
    {
        if (std_dev(i) == 0)
            std_dev(i) = 1.0;
    }
    std::ofstream ofs("scaler_params.bin", std::ios::binary);
    if (!ofs.is_open())
        throw std::runtime_error("Error: Could not open file for writing.");
    int size = static_cast<int>(mean.size());
    ofs.write(reinterpret_cast<const char*>(&size), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(mean.data()), size * sizeof(double));
    ofs.write(reinterpret_cast<const char*>(std_dev.data()), size * sizeof(double));
    ofs.close();
    std::cout << "scaling parameters saved." << std::endl;
}

void save_split_scaler(const std::string& path, size_t val_size)
{
    rapidcsv::Document doc(path, rapidcsv::LabelParams(-1, -1));

    rapidcsv::Document shuffled_doc = shuffle_rows(doc);
    save_split(shuffled_doc, val_size);
    save_scale(shuffled_doc); 
}

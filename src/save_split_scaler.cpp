#include "save_split_scaler.hpp"

rapidcsv::Document shuffle_rows(const rapidcsv::Document& doc)
{

    size_t rowCount = doc.GetRowCount();
    rapidcsv::Document shuffled_doc("", rapidcsv::LabelParams(-1, -1));

    std::vector<size_t> indices(rowCount);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 g(42);
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < rowCount; ++i)
        shuffled_doc.SetRow(i, doc.GetRow<std::string>(indices[i]));

    return shuffled_doc;
}


void oversample_minority(rapidcsv::Document& trainDoc)
{
    std::vector<char> y = trainDoc.GetColumn<char>(1);

    std::vector<size_t> class_M, class_B;

    for (size_t i = 0; i < y.size(); ++i)
    {
        if (y[i] == 'M')
            class_M.push_back(i);
        else
            class_B.push_back(i);
    }

    std::vector<size_t>* minority;
    std::vector<size_t>* majority;

    if (class_M.size() < class_B.size())
    {
        minority = &class_M;
        majority = &class_B;
    }
    else
    {
        minority = &class_B;
        majority = &class_M;
    }

    size_t diff = majority->size() - minority->size();

    std::mt19937 gen(42);
    std::uniform_int_distribution<> dist(0, minority->size() - 1);

    size_t current_size = trainDoc.GetRowCount();

    for (size_t i = 0; i < diff; ++i)
    {
        size_t idx = (*minority)[dist(gen)];
        trainDoc.SetRow(current_size + i, trainDoc.GetRow<std::string>(idx));
    }
}


std::pair<rapidcsv::Document, rapidcsv::Document> train_val_split(const rapidcsv::Document& shuffled_doc, size_t val_size)
{
    std::vector<char> y = shuffled_doc.GetColumn<char>(0);
    std::vector<size_t> class_1, class_2;

    for (size_t i = 0; i < y.size(); ++i)
    {
        if (y[i] == 'M')
            class_1.push_back(i);
        else
            class_2.push_back(i);
    }

    size_t val_size_class_1 = (class_1.size() * val_size) / 100;
    size_t val_size_class_2 = (class_2.size() * val_size) / 100;

    rapidcsv::Document valDoc("", rapidcsv::LabelParams(-1, -1));
    rapidcsv::Document trainDoc("", rapidcsv::LabelParams(-1, -1));

    size_t i         = 0;
    size_t i_class_1 = 0;
    size_t i_class_2 = 0;

    while (i < val_size_class_1)
    {
        valDoc.SetRow(i, shuffled_doc.GetRow<std::string>(class_1[i_class_1++]));
        ++i;
    }
    while (i < val_size_class_2 + val_size_class_1)
    {
        valDoc.SetRow(i, shuffled_doc.GetRow<std::string>(class_2[i_class_2++]));
        ++i;

    }

    size_t size_class1 = class_1.size() - i_class_1;
    size_t size_class2 = class_2.size() - i_class_2;

    i = 0;
    while (i < size_class1) {
        trainDoc.SetRow(i, shuffled_doc.GetRow<std::string>(class_1[i_class_1++]));
        ++i;
    }
    while (i < size_class1 + size_class2) {
        trainDoc.SetRow(i, shuffled_doc.GetRow<std::string>(class_2[i_class_2++]));
        ++i;
    }
    return {trainDoc, valDoc};
}


void    save_scale(rapidcsv::Document& doc)
{
    doc.RemoveColumn(0);
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
    doc.RemoveColumn(0);

    auto train_val_pair = train_val_split(doc, val_size);
    train_val_pair.first.Save("data/data_train.csv");
    train_val_pair.second.Save("data/data_val.csv");

    save_scale(train_val_pair.first);
    std::cout << "train/valid split saved in data/\n";
    
}

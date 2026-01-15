#include "csv_split.hpp"

std::vector<std::string> csv_to_rawstrs(const char* csvpath)
{
    std::string              row;
    std::ifstream            file(csvpath);
    std::vector<std::string> rawdata;

    if (!file.is_open())
        throw std::exception(); // file error exception

    // can reserve here!
    while (getline(file, row))
        rawdata.push_back(row);

    file.close();
    return rawdata;
}

void shuffle_rows(std::vector<std::string>* rowsptr)
{
    std::random_device rd;
    std::mt19937       g(rd());
    std::shuffle(rowsptr->begin(), rowsptr->end(), g);
}

std::pair<std::vector<std::string>, std::vector<std::string>>
split_rows(std::vector<std::string>* rowsptr, float test_size)
{
    std::pair<std::vector<std::string>, std::vector<std::string>> split_pair;

    // if (randomize == true)
    shuffle_rows(rowsptr);

    size_t train_size_ = static_cast<size_t>(rowsptr->size() * test_size);
    size_t i           = 0;
    while (i < rowsptr->size() - train_size_)
    {
        split_pair.first.push_back(rowsptr->at(i));
        ++i;
    }
    size_t j = 0;
    while (j < train_size_)
    {
        split_pair.second.push_back(rowsptr->at(i));
        ++j;
        ++i;
    }
    return split_pair;
}

void save_split_scaled_data(const std::string& path, std::vector<std::string> &rawdata)
{
    auto scaler = Scaler();
    std::pair<MatrixXd, MatrixXd> xy_pair = csv_to_eigen(path);




    // std::ofstream trainf(path + "data_train.csv");
    // std::ofstream valf(path + "data_val.csv");

    // size_t i = 0;
    // while (i < split_data->first.size())
    // {
    //     trainf << split_data->first.at(i) + "\n";
    //     ++i;
    // }
    // i = 0;
    // while (i < split_data->second.size())
    // {
    //     valf << split_data->second.at(i) + "\n";
    //     ++i;
    // }
    // std::cout << "data_train.csv and data_val.csv are saved.\n";
}

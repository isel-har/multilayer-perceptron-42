#include "commands.hpp"

void    print_usage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " split\n"
        << "  " << prog << " train <config.json>\n"
        << "  " << prog << " predict <model_x.bin>\n";
}

int cmd_split()
{
    save_split_scaler("data/", 20);
    return EXIT_SUCCESS;
}

int cmd_train(const char* config_path)
{
    json models_json = load_json(config_path);
    auto scaler = Scaler("scaler_params.bin");
    std::cout << "params loaded\n";
    t_split   datasplit;
    {
        rapidcsv::Document trainDoc("data/data_train.csv", rapidcsv::LabelParams(-1, -1));
        rapidcsv::Document valDoc("data/data_val.csv", rapidcsv::LabelParams(-1, -1));

        std::pair<MatrixXd, MatrixXd> train_pair = csv_to_eigen(trainDoc);
        std::pair<MatrixXd, MatrixXd> val_pair   = csv_to_eigen(valDoc);
        std::cout << "csv to egein passed\n";
        datasplit.X_train = scaler.fit_transform(train_pair.first);
        std::cout << "problem in fit transform\n";
        datasplit.X_val   = scaler.fit_transform(val_pair.first);

        datasplit.y_train = train_pair.second;
        datasplit.y_val   = val_pair.second;
    }
    (void)datasplit;
    // std::vector<MLPClassifier> models;
    // std::vector<History> histories;

    // if (!models_json.size())
    //     models.emplace_back(models_json);
    // else
    //     for (const auto& jmodel : models_json)
    //         models.emplace_back(jmodel);

    // const unsigned int input_shape =
    //     static_cast<unsigned int>(datasplit.X_train.cols());

    // for (size_t i = 0; i < models.size(); ++i)
    // {
    //     std::cout << "model ______________[" << i + 1 << "]______________\n";

    //     models[i].build(input_shape);
    //     histories.push_back(models[i].fit(datasplit));
    //     models[i].save("model_" + std::to_string(i + 1) + ".bin");
    // }

    // std::vector<std::vector<PlotData>>  figures;
    // std::vector<std::string>            ylabels;

    // std::vector<std::string> metrics({"loss", "accuracy"});
    // for (auto& metric : metrics)
    // {
    //     std::vector<PlotData>   plots;
    //     for (size_t i = 0; i < histories.size(); ++i)
    //     {
    //         std::string model_num = std::to_string(i + 1);
    //         plots.emplace_back(metric + " train per epoch model:" + model_num, histories[i].vecMap[metric].first, "solid");
    //         plots.emplace_back(metric + " val per epoch model:" + model_num, histories[i].vecMap[metric].second, "dashed");
    //     }
    //     figures.push_back(plots);
    //     ylabels.push_back(metric);
    // }

    // Visualizer::multi_figures(figures, ylabels);
    // Visualizer::show();

    return EXIT_SUCCESS;
}

int cmd_predict(const char *model_path)
{   
    (void)model_path;
    // MLPClassifier                   model;
    // double                          loss;
    // MatrixXd                        y_preds;
    // std::pair<MatrixXd, MatrixXd>   xy_pair;

    // model = MLPClassifier();
    // model.load(std::string(model_path));

    // xy_pair       = csv_to_eigen("data/data.csv");
    // xy_pair.first = StandardScaler(xy_pair.first);
    // y_preds       = model.predict(xy_pair.first, false);
    // loss          = BinarycrossEntropy().compute(y_preds, xy_pair.second);

    // std::cout << "model loss evaluation :" << loss << "\n";
    return EXIT_SUCCESS;
}
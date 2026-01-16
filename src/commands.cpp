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
    json         models_json = load_json(config_path);
    DatasetSplit datasplit   = train_val_split();


    std::vector<MLPClassifier> models;
    std::vector<History> histories;

    if (!models_json.size())
        models.emplace_back(models_json);
    else
        for (const auto& jmodel : models_json)
            models.emplace_back(jmodel);

    const unsigned int input_shape =
        static_cast<unsigned int>(datasplit.X_train.cols());

    for (size_t i = 0; i < models.size(); ++i)
    {
        std::cout << "model ______________[" << i + 1 << "]______________\n";

        models[i].build(input_shape);
        histories.push_back(models[i].fit(datasplit));
        models[i].save("model_" + std::to_string(i + 1) + ".bin");
    }

    std::vector<std::vector<PlotData>>  figures;
    std::vector<std::string>            ylabels;

    std::vector<std::string> metrics({"loss", "accuracy"});
    for (auto& metric : metrics)
    {
        std::vector<PlotData>   plots;
        for (size_t i = 0; i < histories.size(); ++i)
        {
            std::string model_num = std::to_string(i + 1);
            plots.emplace_back(metric + " train per epoch model:" + model_num, histories[i].vecMap[metric].first, "solid");
            plots.emplace_back(metric + " val per epoch model:" + model_num, histories[i].vecMap[metric].second, "dashed");
        }
        figures.push_back(plots);
        ylabels.push_back(metric);
    }

    Visualizer::multi_figures(figures, ylabels);
    Visualizer::show();

    return EXIT_SUCCESS;
}

int cmd_predict(const char *model_path)
{
    auto model  = MLPClassifier();
    auto scaler = Scaler("scaler_params.bin");
    rapidcsv::Document doc("data/data.csv", rapidcsv::LabelParams(-1, -1));

    MatrixXd  Y_true = doc_to_eigen_encoded(doc);
    doc.RemoveColumn(1);
    MatrixXd  X      = doc_to_eigen(doc);
    scaler.fit_transform(X);
    
    model.load(std::string(model_path));
    auto Y_pred = model.predict(X, false);

    double loss = BinarycrossEntropy().compute(Y_pred, Y_true);
    std::cout << "model loss evaluation :" << loss << "\n";
    return EXIT_SUCCESS;
}
#include "commands.hpp"

void    print_usage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " split <*/file.csv> \n"
        << "  " << prog << " train <config.json>\n"
        << "  " << prog << " predict <model_x.bin>\n";
}

int cmd_split(const char *datapath, const char *val_str)
{
    if (datapath == NULL || val_str == NULL)
    { 
        std::cerr << "data file path and val size required.\n";
        return EXIT_FAILURE;
    }

    try {
        size_t val_size = std::stoul(val_str);
    
        if (!(val_size >= 5 && val_size <= 20)) {
            std::cerr << "validation size between 5 and 20\n";
            return EXIT_FAILURE;
        }
    
        save_split_scaler(datapath, val_size);
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e) {
        std::cerr << "exception caught : " << e.what() << "\n";
    }
    return EXIT_FAILURE;
}

int cmd_train(const char* config_path)
{
    try {

        json         models_json = load_json(config_path);
        DatasetSplit datasplit   = train_val_split();
    
        std::vector<MLPClassifier> models;
        std::vector<History> histories;
    
        std::srand(42);// to change!
    
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
    catch (const std::exception &e) {
        std::cerr << "exception caught : " << e.what() << "\n";
    }
    return EXIT_FAILURE;
}

int cmd_predict(const char* datapath)
{
    if (datapath == NULL)
    {
        std::cerr << "data file path cannot be null.\n";
        return EXIT_FAILURE;
    }
    try {
        rapidcsv::Document doc(datapath, rapidcsv::LabelParams(-1, -1));
        auto scaler = Scaler("scaler_params.bin");
        std::vector<MLPClassifier> models;

        for (const auto& entry : fs::directory_iterator(fs::current_path())) {

            if (!entry.is_regular_file())
                continue;

            fs::path path = entry.path();
            std::string filename = path.filename().string();

            if (path.extension() != ".bin")
                continue;

            if (filename.rfind("model_", 0) != 0)
                continue;
            
            models.emplace_back();
            models.back().load(path);
        }
        
        MatrixXd  Y_true = doc_to_eigen_encoded(doc);
        doc.RemoveColumn(1);
        MatrixXd  X      = doc_to_eigen(doc);
        scaler.transform(X);

        for (size_t i = 0; i < models.size(); ++i) {

            auto Y_pred = models[i].predict(X, false);
            double loss = BinarycrossEntropy().compute(Y_pred, Y_true);
            std::cout << "model "<< i + 1 <<" loss evaluation :" << loss << "\n";
        }
        return EXIT_SUCCESS;
    } 
    catch (const std::exception &e) {
        std::cerr << "exception caught : " << e.what() << "\n";
    }
    return EXIT_FAILURE;
}
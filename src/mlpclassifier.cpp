#include "mlpclassifier.hpp"

std::unordered_map<std::string, Metric*> MLPClassifier::metricsMap = {
    {"accuracy", new Accuracy()},
    {"precision", new Precision()},
    {"loss", new BinarycrossEntropy()}
};

MLPClassifier::MLPClassifier():built(false), loaded(false), confptr(nullptr), earlystopping(false)
{}

MLPClassifier::MLPClassifier(const json& conf) : built(false), loaded(false),earlystopping(false)
{
    this->confptr = &conf;
}

MLPClassifier::~MLPClassifier()
{
    for (size_t i = 0; i < this->metrics.size(); ++i)
    {
        this->metrics[i].second = nullptr;
    }
}

void MLPClassifier::train_val_metrics(unsigned int epoch, const DatasetSplit& dataset, History& history)
{
    MatrixXd ypred_train = this->feed(dataset.X_train);
    MatrixXd ypred_val   = this->feed(dataset.X_val);

    MatrixXd     ypmax_train = this->argmax(ypred_train);
    MatrixXd     ypmax_val   = this->argmax(ypred_val);
    unsigned int index       = epoch - 1;

    std::cout << "epoch " << epoch << '/' << this->epochs;

    double loss     = this->metricsMap["loss"]->compute(ypred_train, dataset.y_train);
    double loss_val = this->metricsMap["loss"]->compute(ypred_val, dataset.y_val);
    history.vecMap["loss"].first[index]  = loss;
    history.vecMap["loss"].second[index] = loss_val;

    for (auto& [name, vec] : history.vecMap)
    {
        if (name != "loss")
        {
            double metric_train     = this->metricsMap[name]->compute(ypmax_train, dataset.y_train);
            double metric_val       = this->metricsMap[name]->compute(ypmax_val, dataset.y_val);
            vec.first.at(index) = metric_train;
            vec.second.at(index) = metric_val;
        }
    }

    std::cout << "- loss:" << loss;
    for (const auto& metric : this->metrics)
    {
        double metric_ = history.vecMap[metric.first].first.at(index);
        std::cout << " - " << metric.first << ':' << metric_;
    }

    std::cout << " | val metric:";
    std::cout << "- loss:" << loss_val;
    for (const auto& metric : this->metrics)
    {
        double metric_ = history.vecMap[metric.first].second.at(index);
        std::cout << " - " << metric.first << ':' << metric_;
    }
    std::cout << std::endl;
}

std::vector<json> MLPClassifier::default_layers()
{
    std::vector<json> jlayers;
    json              hidden;

    hidden["size"]       = 16;
    hidden["activation"] = "relu";
    jlayers.push_back(hidden);
    hidden["size"] = 8;
    jlayers.push_back(hidden);
    hidden["activation"] = "softmax";
    hidden["size"] = 2;
    jlayers.push_back(hidden);
    return jlayers;
}

void MLPClassifier::build(unsigned int shape)
{
    if (this->confptr == nullptr)
        throw std::runtime_error("config object required to build.");

    const json conf = *this->confptr;

    this->input_shape            = shape;
    double      learning_rate    = checked_range(conf.value("learning_rate", 0.01), 0.001, 0.1, "learning_rate");
    this->epochs                 = checked_range(conf.value("epochs", 10), 1, 200, "epochs");
    this->batch_size             = checked_range(conf.value("batch_size", 32), 1, 256, "batch_size"); 
    this->earlystopping._enabled = conf.value("early_stopping", false);

    std::vector<std::string> metrics = conf.value("metrics", std::vector<std::string>({}));
    checked_range(metrics.size(), (size_t)0, (size_t)4, "metrics_size");

    auto unique_metrics = std::unordered_set<std::string>(metrics.begin(), metrics.end());
    
    for (const auto& metric : unique_metrics)
    {
        if (MLPClassifier::metricsMap.find(metric) != MLPClassifier::metricsMap.end())
        {
            this->metrics.push_back(std::make_pair(metric, MLPClassifier::metricsMap[metric]));
        }
    }
    std::vector<json> layers_json = conf.value("hidden_layers", this->default_layers()); // default two hidden layers + output layer
    checked_range(layers_json.size(), (size_t)1, (size_t)11, "layers_stack_size");
    checked_layers(layers_json);
    
    this->layers.emplace_back(layers_json[0], shape);
    for (size_t i = 1; i < layers_json.size(); ++i)
    {
        unsigned int shape_ = layers_json[i - 1]["size"];
        this->layers.emplace_back(layers_json[i], shape_);
    }

    this->layers.emplace_back(layers_json.back()["size"], 2, "softmax", false);

    std::string optimizer_str = conf.value("optimizer", "gd");
    if (optimizer_str == "gd") this->optimizer   = new GradientDescent(learning_rate);
    if (optimizer_str == "adam") this->optimizer = new Adam(learning_rate, this->layers);
    else this->optimizer = new GradientDescent(learning_rate);
    this->built = true;
}

MatrixXd MLPClassifier::feed(const MatrixXd& x)
{
    if (layers[0].weights.rows() != x.cols())
        throw std::runtime_error("input cols not equal to weights rows.");

    MatrixXd feed = layers[0].forward(x);
    for (size_t i = 1; i < this->layers.size(); ++i)
    {
        feed = layers[i].forward(feed);
    }
    return feed;
}

void MLPClassifier::backward(const MatrixXd& dl_out)
{
    int      last  = (int) this->layers.size() - 1;
    MatrixXd dloss = dl_out;
    for (; last >= 0; --last)
    {
        dloss = this->layers[last].backward(dloss);
    }
}

// MatrixXd MLPClassifier::argmax(const MatrixXd& y_probs) const
// {
//     MatrixXd result = MatrixXd::Zero(y_probs.rows(), y_probs.cols());
//     for (size_t i = 0; i < (size_t) y_probs.rows(); ++i)
//     {
//         size_t index     = (y_probs(i, 0) > y_probs(i, 1)) ? 0 : 1;
//         result(i, index) = 1.0f;
//     }
//     return result;
// }

MatrixXd MLPClassifier::argmax(const MatrixXd& y_probs) const
{
    MatrixXd result = MatrixXd::Zero(y_probs.rows(), y_probs.cols());

    auto mask = (y_probs.col(0).array() > y_probs.col(1).array());

    result.col(0) = mask.cast<double>();
    result.col(1) = (!mask).cast<double>();

    return result;
}

History MLPClassifier::fit(const DatasetSplit& dataset)
{
    if (!this->built)
        throw std::runtime_error("build required before training phase.");

    if (this->input_shape != (size_t) dataset.X_train.cols())
        throw std::runtime_error("input shape must be equal to given input cols");

    History history(this->epochs);
    unsigned int e = 1;
    double      loss_e = 0.0;

    while (e <= epochs && !earlystopping(loss_e))
    {
        for (unsigned int i = 0; i < (unsigned int) dataset.X_train.rows(); i += batch_size)
        {
            unsigned int end = std::min(i + batch_size, (unsigned int) dataset.X_train.rows());
    
            MatrixXd xbatch = dataset.X_train.middleRows(i, end - i);
            MatrixXd ybatch = dataset.y_train.middleRows(i, end - i);
    
            MatrixXd probs = this->feed(xbatch);
            MatrixXd loss  = (probs.array() - ybatch.array()).matrix();
    
            this->backward(loss);
            this->optimizer->update(this->layers);
        }
        this->train_val_metrics(e, dataset, history);
        loss_e = history.vecMap["loss"].second[e - 1];
        ++e;
    }
    return history;
}

MatrixXd MLPClassifier::predict(const MatrixXd& x, bool argmaxed = false)
{
    MatrixXd logits = this->feed(x);
    return  argmaxed ? this->argmax(logits) : logits;
}

void MLPClassifier::save(const std::string &name) const
{
    std::ofstream file(name, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open file for saving model.");

    size_t total_layers = this->layers.size();
    file.write(reinterpret_cast<const char*>(&total_layers), sizeof(total_layers));

    for (const auto& layer : layers)
    {
        unsigned int size = layer.size;
        unsigned int input_shape = layer.input_shape;

        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(&input_shape), sizeof(input_shape));

        size_t activation_length = layer.activation__.length();
        file.write(reinterpret_cast<const char*>(&activation_length), sizeof(activation_length));
        file.write(layer.activation__.c_str(), activation_length);

        for (unsigned int i = 0; i < input_shape; ++i)
        {
            for (unsigned int j = 0; j < size; ++j)
            {
                file.write(reinterpret_cast<const char*>(&layer.weights(i, j)), sizeof(double));
            }
        }
    
        for (unsigned int i = 0; i < size; ++i)
        {
            file.write(reinterpret_cast<const char*>(&layer.biases(0, i)), sizeof(double));
        }
    }
    std::cout << '\'' << name + "\' saved\n";
}

void MLPClassifier::safe_read(std::ifstream& file, char* buffer, std::size_t size){
    if (!file.read(buffer, size))
        throw std::runtime_error("Corrupted or truncated model file.");
}


void MLPClassifier::load(const std::string& model_path)
{
    std::ifstream file(model_path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to load the model: " + model_path);

    size_t total_layers;
    MLPClassifier::safe_read(file, reinterpret_cast<char*>(&total_layers), sizeof(total_layers));

    this->layers.reserve(total_layers);

    for (size_t i = 0; i < total_layers; ++i)
    {
        unsigned int size        = 0;
        unsigned int input_shape = 0;
        size_t activation_length = 0;

        MLPClassifier::safe_read(file, reinterpret_cast<char*>(&size), sizeof(size));
        MLPClassifier::safe_read(file, reinterpret_cast<char*>(&input_shape), sizeof(input_shape));
        MLPClassifier::safe_read(file, reinterpret_cast<char*>(&activation_length), sizeof(activation_length));

        if (size == 0 || input_shape == 0 || activation_length > 50)
            throw std::runtime_error("Invalid layer metadata.");


        std::string activation;
        activation.resize(activation_length);
        file.read(&activation[0], activation_length);

        this->layers.emplace_back(input_shape, size, activation, true);
        auto &layer = this->layers.back();

        for (unsigned int i = 0; i < input_shape; ++i)
        {
            for (unsigned int j = 0; j < size; ++j)
            {
                file.read(reinterpret_cast<char*>(&layer.weights(i, j)), sizeof(double));
            }
        }

        for (unsigned int i = 0; i < size; ++i)
        {
            file.read(reinterpret_cast<char*>(&layer.biases(0, i)), sizeof(double));
        }
    }
    std::cout << "'" << model_path << "' loaded\n";
}

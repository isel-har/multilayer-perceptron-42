#pragma once

#include "csv_to_eigen.hpp"
#include "earlystopping.hpp"
#include "history.hpp"
#include "json.hpp"
#include "layer.hpp"
#include "metrics.hpp"
#include "optimizers.hpp"
#include "checker.tpp"
#include "binary_cross_entropy.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <unordered_set>
#include <utility>

using json = nlohmann::json;


class MLPClassifier
{
  private:
  
  static std::unordered_map<std::string, Metric*> metricsMap;
  unsigned int epochs;
  unsigned int batch_size;
  unsigned int input_shape;
  bool         built;


  std::vector<Layer>                           layers;
  
  Optimizer*    optimizer = nullptr;
  const json*   confptr   = nullptr;
  
  EarlyStopping       earlystopping;
  BinaryCrossEntropy  loss;

  MatrixXd feed(const MatrixXd&);
  void     backward(const MatrixXd&, const MatrixXd&);
  
  public:
    std::vector<std::pair<std::string, Metric*>> metrics;
    MLPClassifier();
    MLPClassifier(const json&);
    ~MLPClassifier();

    void    load(const std::string &model_path);
    void    save(const std::string &name) const;
    void    build(unsigned int);
    History fit(const DatasetSplit&);


    void                      set_weights(const std::vector<Layer> &);
    std::vector<Layer>        get_weights() const;

    std::vector<json> default_layers();

    MatrixXd argmax(const MatrixXd&) const;
    MatrixXd predict(const MatrixXd& x, bool argmaxed);

    void train_val_metrics(unsigned int epoch, const DatasetSplit& dataset, History& history);
    static void safe_read(std::ifstream& file, char* buffer, std::size_t size);
    static void clean_static_var();
};

static std::unordered_map<std::string, Metric*> metricsMap;

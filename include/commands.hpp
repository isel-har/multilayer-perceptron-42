#ifndef COMMANDS_HPP
#define COMMANDS_HPP

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>

#include "mlpclassifier.hpp"
#include "json_loader.hpp"
#include "visualizer.hpp"
#include "save_split_scaler.hpp"
#include "metrics.hpp"

namespace fs = std::filesystem;

int     cmd_split(const char* datapath, const char *val_str);
int     cmd_train(const char* config_path);
int     cmd_predict(const char* datapath);
void    print_usage(const char* prog);

#endif

#ifndef SAVE_SPLIT_SCLAER_HPP
#define SAVE_SPLIT_SCLAER_HPP

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include "scaler.hpp"
#include "csv_to_eigen.hpp"
#include <ctime>

rapidcsv::Document shuffle_rows(const rapidcsv::Document& doc);

void    save_scale(rapidcsv::Document& doc);
void    save_split_scaler(const std::string& path, size_t val_size);
void    oversample_minority(rapidcsv::Document& trainDoc);

std::pair<rapidcsv::Document, rapidcsv::Document>    train_val_split(const rapidcsv::Document& doc, size_t val_size);
#endif
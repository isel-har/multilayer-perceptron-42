#ifndef DATA_SPLITER_H
#define DATA_SPLITER_H

#include <utility>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <fstream>

std::pair<std::vector<std::string>, std::vector<std::string>>   split_rows(std::vector<std::string>*, float);
std::vector<std::string>                                        csv_to_rawstrs(const char*h);
void                                                            shuffle_rows(std::vector<std::string>*);
void                                                            save_splitted_data(const std::string &, std::pair<std::vector<std::string>, std::vector<std::string>>*);

#endif
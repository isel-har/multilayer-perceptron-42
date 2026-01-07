#ifndef HISTORY_HPP
#define HISTORY_HPP

#include <vector>
#include <string>
#include <unordered_map>

class History {
public:
    std::vector<double> loss;
    std::vector<double> accuracy;
    std::vector<double> precision;
    std::vector<double> recall;
    History(size_t);

    std::unordered_map<std::string, std::vector<double>*> vecMap;
};

#endif

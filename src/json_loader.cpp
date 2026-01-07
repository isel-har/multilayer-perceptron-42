#include "json_loader.hpp"

json    load_json(const char*path) {

    std::ifstream   inputfile(path);

    if (!inputfile.is_open())
        throw std::runtime_error("Could not open config file.");
        
    return json::parse(inputfile);
}

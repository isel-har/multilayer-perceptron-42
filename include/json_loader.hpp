#ifndef JSON_LOADER_HPP
#define JSON_LOADER_HPP

#include "json.hpp"
#include <fstream>

using json = nlohmann::json;

json    load_json(const char *);

#endif
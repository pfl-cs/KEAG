#include <iostream>
#include <vector>
#include "../config/Config.h"
#ifndef DATA_PROCESS_H
#define DATA_PROCESS_H

void save_meta_info(const std::string& path, std::vector<double>& lbds, std::vector<double>& ubds);
void load_meta_info(const std::string& path, std::vector<double>& lbds, std::vector<double>& ubds);

#endif //DATA_PROCESS_H

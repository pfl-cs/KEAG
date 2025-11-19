#include "data_process.h"
#include <fstream>
#include <cassert>
#include <random>
#include <iostream>
#include <unordered_set>
#include <spatialindex/utils/file_utils.h>
#include <spatialindex/utils/data_utils.h>
#include <spatialindex/SpatialIndex.h>
using namespace SpatialIndex;

void save_meta_info(const std::string& path, std::vector<double>& lbds, std::vector<double>& ubds)
{
    std::ofstream fout(path.c_str(), std::ios::out | std::ios::binary);
    int dim = lbds.size();
    fout.write(reinterpret_cast<char*>(&dim), sizeof(int));
    fout.write(reinterpret_cast<char*>(&lbds[0]), sizeof(double) * dim);
    fout.write(reinterpret_cast<char*>(&ubds[0]), sizeof(double) * dim);
    fout.close();
}

void load_meta_info(const std::string& path, std::vector<double>& lbds, std::vector<double>& ubds)
{
    std::ifstream fin(path.c_str(), std::ios::in | std::ios::binary);
    int dim = -1;
    fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
    // std::cout << "dim = " << dim << std::endl;
    lbds.resize(dim);
    ubds.resize(dim);
    fin.read(reinterpret_cast<char*>(&lbds[0]), sizeof(double) * dim);
    fin.read(reinterpret_cast<char*>(&ubds[0]), sizeof(double) * dim);
    fin.close();
}

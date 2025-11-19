#include <filesystem>
#include <iostream>
#include <vector>
#include <spatialindex/SpatialIndex.h>
#include <spatialindex/RTree.h>
using namespace SpatialIndex;
namespace fs = std::filesystem;

#ifndef MYDATASTREM_H
#define MYDATASTREM_H
class MyDataStream : public IDataStream
{
public:
    int dim;

    bool debug_mode = false;

    MyDataStream(int dim, std::string inputFile) : dim(dim), m_pNext(nullptr)
    {
        m_fin.open(inputFile.c_str(), std::ios::in | std::ios::binary);

        if (!m_fin)
            throw Tools::IllegalArgumentException("Input file not found.");

        readNextEntry();
    }

    MyDataStream(int dim, std::string inputFile, bool _debug_mode) : dim(dim), m_pNext(nullptr), debug_mode(_debug_mode)
    {
        m_fin.open(inputFile.c_str(), std::ios::in | std::ios::binary);

        if (!m_fin)
            throw Tools::IllegalArgumentException("Input file not found.");

        readNextEntry();
    }

    ~MyDataStream() override
    {
        if (m_pNext != nullptr)
            delete m_pNext;
    }

    void set_debug_mode()
    {
        debug_mode = true;
    }

    IData* getNext() override
    {
        if (m_pNext == nullptr)
            return nullptr;

        RTree::Data* ret = m_pNext;
        m_pNext = nullptr;
        readNextEntry();
        return ret;
    }

    bool hasNext() override { return (m_pNext != nullptr); }

    uint32_t size() override
    {
        throw Tools::NotSupportedException("Operation not supported.");
    }

    void rewind() override
    {
        if (m_pNext != nullptr)
        {
            delete m_pNext;
            m_pNext = nullptr;
        }

        m_fin.seekg(0, std::ios::beg);
        readNextEntry();
    }

    void readNextEntry()
    {
        id_type id;
        //        double low[2], high[2];
        std::vector<double> low(dim, 0);
        std::vector<double> high(dim, 0);

        //		m_fin >> id;
        m_fin.read((char*)&id, sizeof(id_type));
        if (debug_mode)
        {
            std::cout << "id = " << id << std::endl;
        }
        for (int i = 0; i < dim; ++i)
        {
            m_fin.read((char*)&(low[i]), sizeof(double));
            //            m_fin >> low[i];
        }
        for (int i = 0; i < dim; ++i)
        {
            m_fin.read((char*)&(high[i]), sizeof(double));
            //            m_fin >> high[i];
        }
        //        m_fin >> low[0] >> low[1] >> high[0] >> high[1];
        if (m_fin.good())
        {
            // if (id == 2 && low[0] < 60 || id == 14 && low[0] > 50)
            // {
            //     std::cout << id << ", " << low[0] << ", " << low[1] << ", " << high[0]
            //         << ", " << high[1] << std::endl;
            // }

            Region r(low.data(), high.data(), dim);
            m_pNext = new RTree::Data(sizeof(double),
                                      reinterpret_cast<uint8_t*>(low.data()), r, id);
            // Associate a bogus data array with every entry for testing purposes.
            // Once the data array is given to RTRee:Data a local copy will be
            // created. Hence, the input data array can be deleted after this
            // operation if not needed anymore.
            if (debug_mode)
            {
                r.print_info();
            }
        }
    }

    std::ifstream m_fin;
    RTree::Data* m_pNext;
};


class MyPayloadDataStream : public IDataStream
{
public:
    int dim;
    bool debug_mode = false;

    MyPayloadDataStream(int dim, std::string inputFile) : dim(dim), m_pNext(nullptr)
    {
        m_fin.open(inputFile.c_str(), std::ios::in | std::ios::binary);

        if (!m_fin)
            throw Tools::IllegalArgumentException("Input file not found.");

        readNextEntry();
    }

    MyPayloadDataStream(int dim, std::string inputFile, bool _debug_mode) : dim(dim), m_pNext(nullptr), debug_mode(_debug_mode)
    {
        m_fin.open(inputFile.c_str(), std::ios::in | std::ios::binary);

        if (!m_fin)
            throw Tools::IllegalArgumentException("Input file not found.");

        readNextEntry();
    }

    ~MyPayloadDataStream() override
    {
        if (m_pNext != nullptr)
            delete m_pNext;
    }

    void set_debug_mode()
    {
        debug_mode = true;
    }

    IData* getNext() override
    {
        if (m_pNext == nullptr)
            return nullptr;

        RTree::Data* ret = m_pNext;
        m_pNext = nullptr;
        readNextEntry();
        return ret;
    }

    bool hasNext() override { return (m_pNext != nullptr); }

    uint32_t size() override
    {
        throw Tools::NotSupportedException("Operation not supported.");
    }

    void rewind() override
    {
        if (m_pNext != nullptr)
        {
            delete m_pNext;
            m_pNext = nullptr;
        }

        m_fin.seekg(0, std::ios::beg);
        readNextEntry();
    }

    void readNextEntry()
    {
        const int payload_size = 8;
        int64_t payload[payload_size];
        std::vector<double> low(dim, 0);
        std::vector<double> high(dim, 0);
        m_fin.read((char*)payload, sizeof(id_type) * payload_size);
        id_type id = payload[0];

        for (int i = 0; i < dim; ++i)
        {
            m_fin.read((char*)&(low[i]), sizeof(double));

        }
        for (int i = 0; i < dim; ++i)
        {
            m_fin.read((char*)&(high[i]), sizeof(double));

        }

        if (m_fin.good())
        {

            Region r(low.data(), high.data(), dim);
            m_pNext = new RTree::Data(sizeof(double),
                                      reinterpret_cast<uint8_t*>(low.data()), r, id);
            // Associate a bogus data array with every entry for testing purposes.
            // Once the data array is given to RTRee:Data a local copy will be
            // created. Hence, the input data array can be deleted after this
            // operation if not needed anymore.
            if (debug_mode)
            {
                r.print_info();
            }
        }
    }

    std::ifstream m_fin;
    RTree::Data* m_pNext;
};



#endif //MYDATASTREM_H

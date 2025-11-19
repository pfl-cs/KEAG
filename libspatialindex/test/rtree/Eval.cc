#include <filesystem>
#include <iostream>
#include <spatialindex/SpatialIndex.h>
#include <spatialindex/RTree.h>
#include <spatialindex/utils/file_utils.h>
#include <spatialindex/utils/vector_utils.h>
#include <unordered_set>
#include <random>
#include <fstream>
#include "../config/Config.h"
#include "../data_process/MyDataStrem.h"
#include "../data_process/data_process.h"
using namespace SpatialIndex;
namespace fs = std::filesystem;

#define INSERT 1
#define DELETE 0
#define QUERY 2


int main(int argc, char* argv[])
{
    Config cfg;
    parse_args(argc, argv, cfg);
    int dim = cfg.dataset.dim;

    std::string init_path = cfg.dataset.test_init_path;
    std::string insert_path = cfg.dataset.test_insert_path;
    std::string queries_path = cfg.dataset.test_queries_path;

    try
    {
        MyDataStream stream_KEAG(dim, init_path);
        MyDataStream stream_rstar(dim, init_path);
        MyDataStream stream_rtreeq(dim, init_path);

        MyDataStream stream_insert(dim, insert_path);

        std::vector<double> lbds_init, ubds_init, lbds_insert, ubds_insert, lbds, ubds;
        load_meta_info(cfg.dataset.test_init_meta_info_path, lbds_init, ubds_init);
        load_meta_info(cfg.dataset.test_insert_meta_info_path, lbds_insert, ubds_insert);

        vector_utils::getSmallerVector(lbds_init, lbds_insert, lbds);
        vector_utils::getLargerVector(ubds_init, ubds_insert, ubds);

        double utilization = 0.9;
        int indexCapacity = 50;
        int leafCapacity = 50;

        IStorageManager* memory_manager_KEAG =
            StorageManager::createNewMemoryStorageManager();
        IStorageManager* memory_manager_rstar =
            StorageManager::createNewMemoryStorageManager();
        IStorageManager* memory_manager_rtreeq =
            StorageManager::createNewMemoryStorageManager();

        // Create and bulk load a new RTree with dimensionality 2, using "file" as
        // the StorageManager and the RSTAR splitting policy.
        id_type indexIdentifier;
        ISpatialIndex* KEAG;

        assert (cfg.model_task == "split" || cfg.model_task == "subtree" || cfg.model_task == "both");
        if (cfg.model_task == "split")
        {
            KEAG = RTree::createAndBulkLoadNewRTree(
              RTree::BLM_STR, stream_KEAG, *memory_manager_KEAG, utilization, indexCapacity,
              leafCapacity, dim, SpatialIndex::RTree::RV_MODEL_SPLIT, indexIdentifier);
        }
        else if (cfg.model_task == "subtree")
        {
            KEAG = RTree::createAndBulkLoadNewRTree(
              RTree::BLM_STR, stream_KEAG, *memory_manager_KEAG, utilization, indexCapacity,
              leafCapacity, dim, SpatialIndex::RTree::RV_MODEL_SUBTREE, indexIdentifier);
        }
        else
        {
            KEAG = RTree::createAndBulkLoadNewRTree(
              RTree::BLM_STR, stream_KEAG, *memory_manager_KEAG, utilization, indexCapacity,
              leafCapacity, dim, SpatialIndex::RTree::RV_MODEL_BOTH, indexIdentifier);
        }

        KEAG->setLbds(lbds);
        KEAG->setUbds(ubds);
        KEAG->setDeviceConfigs();

        if (cfg.model_task == "split" || cfg.model_task == "both")
        {
            KEAG->setSplitNormalizationInfosPath(cfg.split.normalization_infos_path);
            KEAG->setSplitModelPath(cfg.split.jit_module_path);
            KEAG->loadSplitNormalizationInfos();
            KEAG->loadSplitModel();
        }
        if (cfg.model_task == "subtree" || cfg.model_task == "both")
        {
            KEAG->setSubtreeNormalizationInfosPath(cfg.subtree.normalization_infos_path);
            KEAG->setSubtreeModelPath(cfg.subtree.jit_module_path);
            KEAG->loadSubtreeNormalizationInfos();
            KEAG->loadSubtreeModel();
        }

        id_type indexIdentifier_rstar;
        ISpatialIndex* rstar = RTree::createAndBulkLoadNewRTree(
            RTree::BLM_STR, stream_rstar, *memory_manager_rstar, utilization, indexCapacity,
            leafCapacity, dim, SpatialIndex::RTree::RV_RSTAR, indexIdentifier_rstar);

        double utilization_rtree = 0.4;
        id_type indexIdentifier_rtreeq;
        ISpatialIndex* rtreeq = RTree::createAndBulkLoadNewRTree(
            RTree::BLM_STR, stream_rtreeq, *memory_manager_rtreeq, utilization_rtree, indexCapacity,
            leafCapacity, dim, SpatialIndex::RTree::RV_QUADRATIC, indexIdentifier_rtreeq);

        std::string logs_path = "logs.txt";
        std::ofstream log_out(logs_path.c_str());

        std::cout << "KEAG.rootLevel = " << KEAG->getRootLevel() << std::endl;
        std::cout << "KEAG.info:" << std::endl
            << std::endl;
        std::cerr << *KEAG;
        std::cerr << "Index ID of KEAG: " << indexIdentifier << std::endl;
        std::cout << "==============================" << std::endl
            << std::endl;

        log_out << "KEAG.rootLevel = " << KEAG->getRootLevel() << std::endl;
        log_out << "KEAG.info:" << std::endl
            << std::endl;
        log_out << *KEAG;
        log_out << "Index ID of KEAG: " << indexIdentifier << std::endl;
        log_out << "==============================" << std::endl
            << std::endl;

        std::cout << "rstar.info:" << std::endl
            << std::endl;
        std::cerr << *rstar;
        std::cerr << "Index ID of R*-tree: " << indexIdentifier_rstar << std::endl;
        std::cout << "==============================" << std::endl
            << std::endl;

        std::cout << "rtreeq.info:" << std::endl
            << std::endl;
        std::cerr << *rtreeq;
        std::cerr << "Index ID of R-tree(Q): " << indexIdentifier_rtreeq << std::endl;
        std::cout << "==============================" << std::endl
            << std::endl;


        bool ret_KEAG = KEAG->isIndexValid();
        if (ret_KEAG == false)
            std::cerr << "ERROR: KEAG's structure is invalid!" << std::endl;
        else
            std::cerr << "The stucture of KEAG seems O.K." << std::endl;

        bool ret_rstar = rstar->isIndexValid();
        if (ret_rstar == false)
            std::cerr << "ERROR: R*-tree's structure is invalid!" << std::endl;
        else
            std::cerr << "The stucture of R*-tree seems O.K." << std::endl;

        bool ret_rtreeq = rtreeq->isIndexValid();
        if (ret_rtreeq == false)
            std::cerr << "ERROR: R-tree(Q)'s structure is invalid!" << std::endl;
        else
            std::cerr << "The stucture of R-tree(Q) seems O.K." << std::endl;

        MyDataStream _stream_query(dim, queries_path);
        IStorageManager* memory_manager_query =
                    StorageManager::createNewMemoryStorageManager();

        uint64_t init_num_splits = KEAG->getNumSplits();
        uint64_t num_splits = 0;

        uint64_t init_num_splits_rstar = rstar->getNumSplits();
        uint64_t num_splits_rstar = 0;

        uint64_t init_num_splits_rtreeq = rtreeq->getNumSplits();
        uint64_t num_splits_rtreeq = 0;

        uint64_t num_inserts = 0;

        uint64_t delta = 100000ul * cfg.dataset.data_size;

        // uint64_t insert_count = 0;
        std::cout << "Start to insert records" << std::endl;
        while (stream_insert.hasNext())
        {
            RTree::Data* d = reinterpret_cast<RTree::Data*>(stream_insert.getNext());
            if (d == nullptr)
                throw Tools::IllegalArgumentException(
                    "stream_insert expects SpatialIndex::RTree::Data entries."
                );

            KEAG->insertData(0, 0, d->m_region, d->m_id);
            rstar->insertData(0, 0, d->m_region, d->m_id);
            rtreeq->insertData(0, 0, d->m_region, d->m_id);

            d->m_pData = nullptr;
            delete d;

            uint64_t curr_num_splits = KEAG->getNumSplits() - init_num_splits;

            if (curr_num_splits > num_splits) {
                num_splits = curr_num_splits;
            }
            ++num_inserts;

            if (num_inserts % delta == 0)
            {
                std::cout << "data_size = " << cfg.dataset.data_size << ", " << num_inserts << " records have been inserted." << std::endl;
            }
        }

        num_splits_rstar = rstar->getNumSplits() - init_num_splits_rstar;
        num_splits_rtreeq = rtreeq->getNumSplits() - init_num_splits_rtreeq;

        // std::cout << "num_splits_KEAG = " << num_splits << std::endl;
        // std::cout << "num_splits_rstar = " << num_splits_rstar << std::endl;
        // std::cout << "num_splits_rtreeq = " << num_splits_rtreeq << std::endl;
        // std::cout << "num_inserts = " << num_inserts << std::endl;

        // log_out << "num_splits = " << num_splits << std::endl;
        // log_out << "num_splits_rstar = " << num_splits_rstar << std::endl;
        // log_out << "num_splits_rtreeq = " << num_splits_rtreeq << std::endl;
        // log_out << "num_inserts = " << num_inserts << std::endl;

        std::cout << "------------------------------" << std::endl;
        std::cout << "KEAG.info:" << std::endl
                    << std::endl;
        std::cerr << *KEAG;
        std::cerr << "Index ID of KEAG: " << indexIdentifier << std::endl;
        std::cout << "==============================" << std::endl
            << std::endl;

        log_out << "------------------------------" << std::endl;
        log_out << "KEAG.info:" << std::endl
                    << std::endl;
        log_out << *KEAG;
        log_out << "Index ID of KEAG: " << indexIdentifier << std::endl;
        log_out << "==============================" << std::endl
            << std::endl;

        std::cout << "rstar.info:" << std::endl
            << std::endl;
        std::cerr << *rstar;
        std::cerr << "Index ID of Rstar: " << indexIdentifier_rstar << std::endl;
        std::cout << "==============================" << std::endl
            << std::endl;

        std::cout << "rtreeq.info:" << std::endl
            << std::endl;
        std::cerr << *rtreeq;
        std::cerr << "Index ID of R-tree(Q): " << indexIdentifier_rtreeq << std::endl;
        std::cout << "==============================" << std::endl
            << std::endl;

        std::cout << "------------------------------" << std::endl;

        MyDataStream stream_query(dim, queries_path);
        uint64_t net_num_io = 0;
        uint64_t net_num_io_rstar = 0;
        uint64_t net_num_io_rtreeq = 0;
        std::cout << "Start to run queries." << std::endl;
        uint64_t query_no = 0;

        const int num_methods = 3;
        std::vector<uint64_t> num_better_each_method(num_methods, 0);

        while (stream_query.hasNext())
        {
            RTree::Data* d = reinterpret_cast<RTree::Data*>(stream_query.getNext());
            if (d == nullptr)
                throw Tools::IllegalArgumentException(
                    "stream_query expects SpatialIndex::RTree::Data entries."
                );

            uint64_t curr_num_io = KEAG->getIntersectsWithQueryIOCost(d->m_region);
            uint64_t curr_num_io_rstar = rstar->getIntersectsWithQueryIOCost(d->m_region);
            uint64_t curr_num_io_rtreeq = rtreeq->getIntersectsWithQueryIOCost(d->m_region);

            std::vector<uint64_t> curr_ios = {curr_num_io, curr_num_io_rstar};
            std::vector<int64_t> best_idxes;
            vector_utils::argmin(curr_ios, best_idxes);
            for (int64_t best_idx : best_idxes)
            {
                ++num_better_each_method[best_idx];
            }

            net_num_io += curr_num_io;
            net_num_io_rstar += curr_num_io_rstar;
            net_num_io_rtreeq += curr_num_io_rtreeq;

            ++query_no;
            if (query_no % 1000 == 0)
            {
                std::cout << "Query-" << query_no << " has been executed. net_num_io = " << net_num_io << ", net_num_io_rstar = " << net_num_io_rstar << ", net_num_io_rtreeq = " << net_num_io_rtreeq << std::endl;
                // std::cout << "\t num_better = " << num_better_each_method[0] << ", num_better_rstar = " << num_better_each_method[1] << ", num_better_rtreeq = " << num_better_each_method[2] << std::endl;
            }

            delete d;
        }

        std::cout << "net_num_io_KEAG = " << net_num_io << std::endl;
        std::cout << "net_num_io_rstar = " << net_num_io_rstar << std::endl;
        std::cout << "net_num_io_rtreeq = " << net_num_io_rtreeq << std::endl;

        log_out << "net_num_io_KEAG = " << net_num_io << std::endl;
        log_out << "net_num_io_rstar = " << net_num_io_rstar << std::endl;
        log_out << "net_num_io_rtreeq = " << net_num_io_rtreeq << std::endl;
        log_out.close();

        delete KEAG;
        delete rstar;
        delete rtreeq;
        delete memory_manager_KEAG;
        delete memory_manager_rstar;
        delete memory_manager_rtreeq;
        delete memory_manager_query;
    }
    catch (Tools::Exception& e)
    {
        std::cerr << "******ERROR******" << std::endl;
        std::string s = e.what();
        std::cerr << s << std::endl;
        return -1;
    }

    return 0;
}

// ./Eval -d GAU -s 2 -mt split -ua true

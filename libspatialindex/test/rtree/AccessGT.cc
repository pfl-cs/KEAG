#include <filesystem>
#include <iostream>
#include <spatialindex/SpatialIndex.h>
#include <spatialindex/RTree.h>
#include <spatialindex/utils/file_utils.h>
#include <spatialindex/utils/vector_utils.h>
#include <unordered_set>
#include <random>
#include <fstream>
#include <chrono>
#include "../config/Config.h"
#include "../data_process/MyDataStrem.h"
#include "../data_process/data_process.h"

using namespace SpatialIndex;
namespace fs = std::filesystem;
using namespace std::chrono;
#define INSERT 1
#define DELETE 0
#define QUERY 2


int main(int argc, char* argv[])
{
    Config cfg;
    parse_args(argc, argv, cfg);
    int dim = cfg.dataset.dim;

    std::string init_path = cfg.dataset.train_init_path;
    std::string insert_path = cfg.dataset.train_insert_path;
    std::string queries_path = cfg.dataset.train_queries_path;
    
    try
    {
        MyDataStream stream(dim, init_path);
        MyDataStream stream_insert(dim, insert_path);

        std::vector<double> lbds_init, ubds_init, lbds_insert, ubds_insert, lbds, ubds;
        load_meta_info(cfg.dataset.train_init_meta_info_path, lbds_init, ubds_init);
        load_meta_info(cfg.dataset.train_insert_meta_info_path, lbds_insert, ubds_insert);

        vector_utils::getSmallerVector(lbds_init, lbds_insert, lbds);
        vector_utils::getLargerVector(ubds_init, ubds_insert, ubds);

        double utilization = 0.9;
        int indexCapacity = 50;
        int leafCapacity = 50;

        IStorageManager* memory_manager =
            StorageManager::createNewMemoryStorageManager();
        IStorageManager* memory_manager_insert =
            StorageManager::createNewMemoryStorageManager();

        id_type indexIdentifier;
        ISpatialIndex* tree_gt;
        assert (cfg.gt_task == "split" || cfg.gt_task == "subtree");
        if (cfg.gt_task == "split")
        {
            tree_gt = RTree::createAndBulkLoadNewRTree(
              RTree::BLM_STR, stream, *memory_manager, utilization, indexCapacity,
              leafCapacity, dim, SpatialIndex::RTree::RV_BEST_SPLIT, indexIdentifier);
        }
        else
        {
            tree_gt = RTree::createAndBulkLoadNewRTree(
              RTree::BLM_STR, stream, *memory_manager, utilization, indexCapacity,
              leafCapacity, dim, SpatialIndex::RTree::RV_BEST_SUBTREE, indexIdentifier);
        }

        tree_gt->setLbds(lbds);
        tree_gt->setUbds(ubds);

        std::cout << "==============================" << std::endl;
        std::cout << "tree.rootLevel = " << tree_gt->getRootLevel() << std::endl;
        std::cout << "tree.info:" << std::endl
            << std::endl;
        std::cerr << *tree_gt;
        std::cerr << "Index ID of Tree: " << indexIdentifier << std::endl;
        std::cout << "==============================" << std::endl
            << std::endl;

        bool ret_gt = tree_gt->isIndexValid();
        if (ret_gt == false)
            std::cerr << "ERROR: Structure is invalid!" << std::endl;
        else
            std::cerr << "The stucture seems O.K." << std::endl;


        MyDataStream _stream_query(dim, queries_path);
        IStorageManager* memory_manager_query =
                    StorageManager::createNewMemoryStorageManager();

        id_type indexIdentifier_query_tree;
        ISpatialIndex* query_tree = RTree::createAndBulkLoadNewRTree(
            RTree::BLM_STR, _stream_query, *memory_manager_query, utilization, indexCapacity,
            leafCapacity, dim, SpatialIndex::RTree::RV_RSTAR, indexIdentifier_query_tree);

        tree_gt->setQueryTree(query_tree);

        bool ret_query = query_tree->isIndexValid();
        if (ret_query == false)
            std::cerr << "ERROR: Query tree's Structure is invalid!" << std::endl;
        else
            std::cerr << "The stucture of the query tree seems O.K." << std::endl;

        std::cout << "query_tree.rootLevel = " << query_tree->getRootLevel() << std::endl;

        std::cout << "query_tree.info:" << std::endl
            << std::endl;
        std::cerr << *query_tree;
        std::cerr << "Index ID of query_tree: " << indexIdentifier_query_tree << std::endl;
        std::cout << "==============================" << std::endl
            << std::endl;

        if (cfg.gt_task == "split")
        {
            tree_gt->turnOnSplitRecording();
            tree_gt->initSplitRecording(cfg.split.split_records_path);
        }
        else
        {
            tree_gt->turnOnSubtreeRecording();
            tree_gt->initSubtreeRecording(cfg.subtree.subtree_records_path);
        }

        uint64_t init_num_splits = tree_gt->getNumSplits();
        uint64_t num_splits = 0;
        uint64_t num_inserts = 0;

        uint64_t delta = 100000ul * cfg.dataset.data_size;

        std::cout << "Start to insert records" << std::endl;
        while (stream_insert.hasNext())
        {
            RTree::Data* d = reinterpret_cast<RTree::Data*>(stream_insert.getNext());
            if (d == nullptr)
                throw Tools::IllegalArgumentException(
                    "stream_insert expects SpatialIndex::RTree::Data entries."
                );

            tree_gt->insertData(0, 0, d->m_region, d->m_id);
            d->m_pData = nullptr;
            delete d;
            uint64_t curr_num_splits = tree_gt->getNumSplits() - init_num_splits;

            if (curr_num_splits > num_splits) {
                num_splits = curr_num_splits;
            }

            ++num_inserts;
            if (num_inserts % delta == 0)
            {
                std::cout << "data_size = " << cfg.dataset.data_size << ", " << num_inserts << " records have been inserted." << std::endl;
            }
        }
        std::cout << "num_splits = " << num_splits << std::endl;
        std::cout << "num_inserts = " << num_inserts << std::endl;
        
        std::cout << "==============================" << std::endl;
        std::cout << "tree.info:" << std::endl
                    << std::endl;
        std::cerr << *tree_gt;
        std::cerr << "Index ID of Tree: " << indexIdentifier << std::endl;
        std::cout << "==============================" << std::endl;

        tree_gt->turnOffSplitRecording();
        tree_gt->turnOffSubtreeRecording();
        if (cfg.gt_task == "split")
        {
            tree_gt->stopSplitRecording();
        }
        else
        {
            tree_gt->stopSubtreeRecording();
        }

        delete tree_gt;
        delete query_tree;
        delete memory_manager;
        delete memory_manager_insert;
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
// ./AccessGT -d GAU -s 2 -gt split

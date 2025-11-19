#include <iostream>
#include <cstdint>
#ifndef CONFIG_H
#define CONFIG_H
typedef float spaceUnit;

struct Config
{
    struct datasetConfig
    {
        std::string name = "GAU";
        std::string data_dir = "";
        int dim = 2;
        int data_size = 25;
        bool use_across = false;

        std::string train_init_path = "";
        std::string train_insert_path = "";
        std::string train_init_meta_info_path = "";
        std::string train_insert_meta_info_path = "";
        std::string train_queries_path = "";

        std::string test_init_path = "";
        std::string test_insert_path = "";
        std::string test_init_meta_info_path = "";
        std::string test_insert_meta_info_path = "";
        std::string test_queries_path = "";
    };

    struct splitConfig
    {
        std::string split_records_path = "";
        bool append_splits = false;
        bool recording_splits = false;
        std::string model_type = "split";
        std::string normalization_infos_path = "";
        std::string ckpt_dir = "";
        std::string jit_module_path = "";
    };

    struct subtreeConfig
    {
        std::string subtree_records_path = "";
        bool append_subtrees = false;
        bool recording_subtrees = false;
        std::string model_type = "subtree";

        std::string normalization_infos_path = "";
        std::string ckpt_dir = "";
        std::string jit_module_path = "";
    };

    std::string project_root = "";
    std::string data_root = "";

    datasetConfig dataset;
    splitConfig split;
    subtreeConfig subtree;


    std::string model_task = "none"; // none, split, subtree, or both
    std::string gt_task = "split"; // none, split, subtree, or both
    std::string model = "KEAG"; // KEAG, simple

    std::string query_type = "range"; // range or knn

    void set_project_root(const std::string& _project_root);
    void finalize();
};

void usage(const std::string& program_name);
void parse_args(int argc, char* argv[], Config& cfg);

#endif //CONFIG_H

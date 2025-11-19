#include "Config.h"
#include <spatialindex/utils/string_utils.h>
#include <spatialindex/utils/file_utils.h>
#include <cmath>

void Config::set_project_root(const std::string& _project_root)
{
    project_root = _project_root;
    data_root = file_utils::path_join(project_root, "data");
}


void Config::finalize()
{
    if (dataset.dim > 2)
    {
        dataset.name += std::to_string(dataset.dim) + "D";
    }

    std::string data_info = dataset.name + "/" + std::to_string(dataset.data_size) + "M";
    dataset.data_dir = file_utils::path_join(data_root, data_info);

    dataset.train_init_path = file_utils::path_join(dataset.data_dir, "train_init.dat");
    dataset.train_insert_path = file_utils::path_join(dataset.data_dir, "train_insert.dat");
    dataset.train_init_meta_info_path = file_utils::path_join(dataset.data_dir, "train_init_meta_info.dat");
    dataset.train_insert_meta_info_path = file_utils::path_join(dataset.data_dir, "train_insert_meta_info.dat");
    dataset.train_queries_path = file_utils::path_join(dataset.data_dir, "train_queries.dat");

    dataset.test_init_path = file_utils::path_join(dataset.data_dir, "test_init.dat");
    dataset.test_insert_path = file_utils::path_join(dataset.data_dir, "test_insert.dat");
    dataset.test_init_meta_info_path = file_utils::path_join(dataset.data_dir, "test_init_meta_info.dat");
    dataset.test_insert_meta_info_path = file_utils::path_join(dataset.data_dir, "test_insert_meta_info.dat");
    dataset.test_queries_path = file_utils::path_join(dataset.data_dir, "test_queries.dat");

    if (dataset.use_across)
    {
        data_info = dataset.name + "/across";
    }

    std::string split_records_fname = "split_records.bin";
    std::string subtree_records_fname = "subtree_records.bin";

    std::string norm_tag = "";
    if (dataset.use_across)
    {
        norm_tag += "across_";
    }
    std::string split_normalization_infos_fname = norm_tag + "split_norm_infos.bin";
    std::string subtree_normalization_infos_fname = norm_tag + "subtree_norm_infos.bin";

    if (model != "KEAG")
    {
        split.model_type = model + "_" + split.model_type;
        subtree.model_type = model + "_" + subtree.model_type;
    }
    split.split_records_path = file_utils::path_join(dataset.data_dir, split_records_fname);
    split.normalization_infos_path = file_utils::path_join(dataset.data_dir, split_normalization_infos_fname);
    split.ckpt_dir = file_utils::path_join(project_root, "ckpt/split");
    split.ckpt_dir = file_utils::path_join(split.ckpt_dir, data_info);
    split.jit_module_path = file_utils::path_join(split.ckpt_dir,split.model_type + ".ts");

    if (!split.recording_splits)
    {
        split.append_splits = false;
    }

    subtree.subtree_records_path = file_utils::path_join(dataset.data_dir, subtree_records_fname);
    subtree.normalization_infos_path = file_utils::path_join(dataset.data_dir, subtree_normalization_infos_fname);
    subtree.ckpt_dir = file_utils::path_join(project_root, "ckpt/subtree");
    subtree.ckpt_dir = file_utils::path_join(subtree.ckpt_dir, data_info);
    subtree.jit_module_path = file_utils::path_join(subtree.ckpt_dir, subtree.model_type + ".ts");

    if (!subtree.recording_subtrees)
    {
        subtree.append_subtrees = false;
    }
}


void usage(const std::string& program_name)
{
    std::cout << "   Usage\t\t\t: " << program_name << std::endl;
    std::cout << "   -h[elp]\t\t\t: Print this help menu." << std::endl;
    std::cout << "   -d[ata]\t\t\t: Data (GAU, Zipf, etc)." << std::endl;
    std::cout << "   -dim\t\t\t\t: Data dimension" << std::endl;
    std::cout << "   -s[ize]\t\t\t: Data size (in millions)." << std::endl;
    std::cout << "   -gt(ground_truth_task)\t\t: Brute force best strategy for none (default), split, subtree or both tasks." << std::endl;
    std::cout << "   -mt(model_task)\t\t: Strategy with ML model for none (default), split, subtree or both tasks." << std::endl;
    std::cout << "   -rsp(recording_splits)\t: true or false. Recording splits or not" << std::endl;
    std::cout << "   -asp(append_splits)\t\t: true or false. Append splits or not" << std::endl;
    std::cout << "   -rsu(recording_subtrees)\t: true or false. Recording subtrees or not" << std::endl;
    std::cout << "   -asu(append_subtrees)\t: true or false. Append subtrees or not" << std::endl;
    std::cout << "   -ua(use_across)\t\t: true or false." << std::endl;
    std::cout << "   -qt(query_type)\t\t: range or knn." << std::endl;
    std::cout << "   -m(model)\t\t\t: standard, simple or simplest." << std::endl;
}

void parse_args(int argc, char* argv[], Config& cfg)
{
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path project_root_path = current_path.parent_path().parent_path().parent_path();
    std::string project_root = project_root_path.string();
    std::cout << "project_root = " << project_root << std::endl;

    cfg.set_project_root(project_root);

    for (int i = 1; i < argc; ++i)
    {
        if (argv[i][0] != '-')
        {
            usage(argv[0]);
            exit(1);
        }
        std::string param(argv[i]);
        if (param.size() <= 1)
        {
            fprintf(stderr, "Error: Invalid command line parameter, %c\n",
                    argv[i][1]);
            usage(argv[0]);
            exit(1);
        }
        param = param.substr(1, param.size() - 1);
        string_utils::str_self_tolower(param);

        if (param == "h" || param == "help")
        {
            usage(argv[0]);
            exit(1);
            break;
        }

        std::string value(argv[++i]);
        string_utils::str_self_tolower(value);

        if (param == "d" || param == "data")
        {
            cfg.dataset.name = value;
        }
        else if (param == "dim")
        {
            int dim = std::stoi(value);
            if (dim >= 2 && dim < 10)
            {
                cfg.dataset.dim = dim;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                       argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "s" || param == "size")
        {
            if (!(string_utils::is_integer(value)))
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
            else
            {
                cfg.dataset.data_size = std::stoi(value);
            }
        }
        else if (param == "asp" || param == "append_splits")
        {
            if (value == "true")
            {
                cfg.split.append_splits = true;
            }
            else if (value == "false")
            {
                cfg.split.append_splits = false;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "rsp" || param == "recording_splits")
        {
            if (value == "true")
            {
                cfg.split.recording_splits = true;
            }
            else if (value == "false")
            {
                cfg.split.recording_splits = false;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "asu" || param == "append_subtrees")
        {
            if (value == "true")
            {
                cfg.subtree.append_subtrees = true;
            }
            else if (value == "false")
            {
                cfg.subtree.append_subtrees = false;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "rsu" || param == "recording_subtrees")
        {
            if (value == "true")
            {
                cfg.subtree.recording_subtrees = true;
            }
            else if (value == "false")
            {
                cfg.subtree.recording_subtrees = false;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "gt" || param == "ground_truth_task")
        {
            if (value == "none" || value == "split" || value == "subtree" || value == "both")
            {
                cfg.gt_task = value;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "mt" || param == "model_task")
        {
            if (value == "none" || value == "split" || value == "subtree" || value == "both")
            {
                cfg.model_task = value;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "ua" || param == "use_across")
        {
            if (value == "true")
            {
                cfg.dataset.use_across = true;
            }
            else if (value == "false")
            {
                cfg.dataset.use_across = false;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "m" || param == "model")
        {
            if (value == "standard" || value == "wophi" || value == "wopoint")
            {
                cfg.model = value;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else if (param == "qt" || param == "query_type")
        {
            if (value == "range" || value == "knn")
            {
                cfg.query_type = value;
            }
            else
            {
                fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
                usage(argv[0]);
                exit(1);
            }
        }
        else
        {
            fprintf(stderr, "Error: Invalid command line parameter, %s\n",
                    argv[i]);
            usage(argv[0]);
            exit(1);
        }
    }
    cfg.finalize();
}
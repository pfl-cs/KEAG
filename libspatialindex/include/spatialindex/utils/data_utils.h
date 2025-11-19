#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <iostream>
#include <cassert>
#include <vector>

namespace data_utils
{
    enum DATA_FLAG
    {
        INT32_FLAG = 0,
        UINT32_FLAG = 1,
        INT64_FLAG = 2,
        UINT64_FLAG = 3,
        FLOAT_FLAG = 4,
        DOUBLE_FLAG = 5
    };

    int64_t load_header(const char *data_path, int &data_type, int &dim,
                        std::vector<int64_t> &shape);

    void load_data(const char *data_path, int data_type, int dim,
                   int64_t act_data_size, void *data);

    int64_t load_float_ndarray(const char *data_path, std::vector<int64_t> &shape,
                               std::vector<float> &array);

    //    template<class T>
    //    int64_t load_ndarray(const char *data_path, std::vector<int64_t> &shape,
    //    std::vector<T> &array);

    // no type check now
    template <class T>
    int64_t load_ndarray(const char *data_path, std::vector<int64_t> &shape,
                         std::vector<T> &array)
    {
        int data_type = -1;
        int dim = -1;
        int64_t act_data_size = load_header(data_path, data_type, dim, shape);
        //        std::cout << "data_type = " << data_type << std::endl;
        array.resize(act_data_size);

        load_data(data_path, data_type, dim, act_data_size, array.data());

        return act_data_size;
    }

    template <class T>
    int64_t save_ndarray(const char *data_path, int data_type, int dim,
                         std::vector<int64_t> &shape, std::vector<T> &data)
    {
        assert(sizeof(int) == 4);
        assert(sizeof(long) == 8);
        assert(sizeof(float) == 4);
        assert(sizeof(double) == 8);
        int64_t act_data_size = 1;
        for (auto x : shape)
        {
            act_data_size *= x;
        }
        assert(act_data_size == data.size());
        FILE *fp = NULL;

        if (NULL == (fp = fopen(data_path, "wb")))
        {
            printf("%s cannot be opened.\n", data_path);
            exit(1);
        }

        int64_t rc = fwrite(&data_type, sizeof(int), 1, fp);
        rc = fwrite(&dim, sizeof(int), 1, fp);
        rc = fwrite(shape.data(), sizeof(int64_t), shape.size(), fp);

        rc = fwrite(data.data(), sizeof(T), data.size(), fp);
        fclose(fp);

        return act_data_size;
    }
} // namespace data_utils

#endif //DATA_UTILS_H

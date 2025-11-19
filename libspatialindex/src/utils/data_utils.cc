#include <spatialindex/utils/data_utils.h>

namespace data_utils
{
    int64_t load_header(const char *data_path, int &data_type, int &dim, std::vector<int64_t> &shape)
    {
        assert(sizeof(int) == 4);
        assert(sizeof(long) == 8);
        assert(sizeof(float) == 4);
        assert(sizeof(double) == 8);
        FILE *fp = NULL;

        if (NULL == (fp = fopen(data_path, "rb")))
        {
            printf("%s cannot be opened.\n", data_path);
            exit(1);
        }

        int64_t rc = fread(&data_type, sizeof(int), 1, fp);

        rc = fread(&dim, sizeof(int), 1, fp);
        assert(dim > 0);

        int64_t act_data_size = 1;
        for (auto i = 0; i < dim; ++i)
        {
            int64_t shape_i = 0;
            rc = fread(&shape_i, sizeof(int64_t), 1, fp);
            shape.emplace_back(shape_i);
            act_data_size *= shape_i;
        }

        fclose(fp);

        return act_data_size;
    }

    void load_data(const char *data_path, int data_type, int dim,
                   int64_t act_data_size, void *data)
    {
        long offset = sizeof(int) * 2 + sizeof(long) * dim;
        FILE *fp = NULL;

        if (NULL == (fp = fopen(data_path, "rb")))
        {
            printf("%s cannot be opened.\n", data_path);
            exit(1);
        }
        fseek(fp, offset, SEEK_SET);

        auto _size = sizeof(int);

        switch (data_type)
        {
        case INT32_FLAG:
            _size = sizeof(int);
            break;
        case UINT32_FLAG:
            _size = sizeof(unsigned int);
            break;
        case INT64_FLAG:
            _size = sizeof(long);
            break;
        case UINT64_FLAG:
            _size = sizeof(unsigned long);
            break;
        case FLOAT_FLAG:
            _size = sizeof(float);
            break;
        case DOUBLE_FLAG:
            _size = sizeof(double);
            break;
        }

        long rc = fread(data, _size, act_data_size, fp);
        assert(rc == act_data_size);
        fclose(fp);
    }

    int64_t load_float_ndarray(const char *data_path, std::vector<int64_t> &shape,
                               std::vector<float> &array)
    {
        int data_type = -1;
        int dim = -1;
        int64_t act_data_size = load_header(data_path, data_type, dim, shape);
        assert(data_type == FLOAT_FLAG);
        array.resize(act_data_size);

        load_data(data_path, data_type, dim, act_data_size, array.data());

        return act_data_size;
    }

} // namespace data_utils

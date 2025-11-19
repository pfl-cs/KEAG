#ifndef VECTOR_UTILS_H
#define VECTOR_UTILS_H
#include <iostream>
#include <vector>
#include <string>

namespace vector_utils
{
    template <class T>
    void getSmallerVector(std::vector<T> &vec1, std::vector<T> &vec2, std::vector<T> &result)
    {
        if (vec1.size() != vec2.size())
        {
            throw std::invalid_argument("vec1.size() != vec2.size()");
        }
        result.clear();
        for (size_t i = 0; i < vec1.size(); ++i)
        {
            result.emplace_back(std::min(vec1[i], vec2[i]));
        }
    }

    template <class T>
    void getLargerVector(std::vector<T> &vec1, std::vector<T> &vec2, std::vector<T> &result)
    {
        if (vec1.size() != vec2.size())
        {
            throw std::invalid_argument("vec1.size() != vec2.size()");
        }
        result.clear();
        for (size_t i = 0; i < vec1.size(); ++i)
        {
            result.emplace_back(std::max(vec1[i], vec2[i]));
        }
    }

    template <class T>
    int64_t argmax(std::vector<T> &vec)
    {
        if (vec.size() == 0)
        {
            return -1;
        }
        int64_t max_idx = 0;
        T max_val = vec[0];
        for (int64_t i = 1; i < vec.size(); ++i)
        {
            if (vec[i] > max_val)
            {
                max_val = vec[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    template <class T>
    int64_t argmin(std::vector<T> &vec)
    {
        if (vec.size() == 0)
        {
            return -1;
        }
        int64_t min_idx = 0;
        T min_val = vec[0];
        for (int64_t i = 1; i < vec.size(); ++i)
        {
            if (vec[i] < min_val)
            {
                min_val = vec[i];
                min_idx = i;
            }
        }
        return min_idx;
    }

    template <class T>
    void argmin(std::vector<T> &vec, std::vector<int64_t> &result)
    {
        if (vec.size() == 0)
        {
            return;
        }
        int64_t min_idx = 0;
        T min_val = vec[0];
        for (int64_t i = 1; i < vec.size(); ++i)
        {
            if (vec[i] < min_val)
            {
                min_val = vec[i];
                min_idx = i;
            }
        }
        for (int64_t i = min_idx; i < vec.size(); ++i)
        {
            if (vec[i] == min_val)
            {
                result.emplace_back(i);
            }
        }
    }

    template <class T>
    std::string tostring(std::vector<T> &vec)
    {
        if (vec.size() == 0)
        {
            return "";
        }
        std::string res = std::to_string(vec[0]);
        for (int64_t i = 1; i < vec.size(); ++i)
        {
            res += ", " + std::to_string(vec[i]);
        }
        return res;
    }

    std::string tostring(std::vector<double> &vec, bool set_precision)
    {
        if (vec.size() == 0)
        {
            return "";
        }
        if (set_precision)
        {
            char buffer[32];
            snprintf(buffer, 30, "%.4f", vec[0]);
            std::string res(buffer);
            for (int64_t i = 1; i < vec.size(); ++i)
            {
                char buffer_i[32];
                snprintf(buffer_i, 30, "%.4f", vec[i]);
                std::string s(buffer_i);
                res += ", " + s;
            }
            return res;
        }
        else
        {
            std::string res = std::to_string(vec[0]);
            for (int64_t i = 1; i < vec.size(); ++i)
            {
                res += ", " + std::to_string(vec[i]);
            }
            return res;
        }
    }
}

#endif //VECTOR_UTILS_H

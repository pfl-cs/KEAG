#ifndef PRINT_UTILS_H
#define PRINT_UTILS_H
#include <iostream>
#include <unordered_set>

namespace print_utils
{
    template <class T>
    void print_vec(const std::vector<T> &vec, const std::string &info)
    {
        std::cout << info << " = [";
        for (size_t i = 0; i < vec.size() - 1; ++i)
        {
            std::cout << vec[i] << ", ";
        }
        std::cout << vec.back() << "]" << std::endl;
    }

    template <class T>
    void print_vec(const std::vector<T> &vec)
    {
        print_vec(vec, "vec");
    }

    template <class T>
    void print_vec_only(const std::vector<T> &vec, char left_bracket = '[')
    {
        char right_bracket = ']';
        if (left_bracket == '{')
        {
            right_bracket = '}';
        }
        std::cout << left_bracket;
        for (size_t i = 0; i < vec.size() - 1; ++i)
        {
            std::cout << vec[i] << ", ";
        }
        std::cout << vec.back() << right_bracket;
    }

    template <class T>
    void print_unordered_set(const std::unordered_set<T> &set, bool sort = true)
    {
        if (sort)
        {
            std::vector<T> vec(set.begin(), set.end());
            std::sort(vec.begin(), vec.end());
            print_vec_only(vec, '{');
        }
        else
        {
            std::string s = "{";
            for (auto it = set.begin(); it != set.end(); ++it)
            {
                std::cout << s << *it;
                s = ", ";
            }
            std::cout << "}";
        }
    }

    template <class T, class S>
    void print_pair_vector(const std::vector<std::pair<T, S>> &vec,
                           const std::string &info)
    {
        std::cout << info << " = [";
        size_t i;
        for (i = 0; i < vec.size() - 1; ++i)
        {
            std::cout << "(" << vec[i].first << ", " << vec[i].second << "), ";
        }
        std::cout << "(" << vec[i].first << ", " << vec[i].second << ")] "
                  << std::endl;
    }
} // namespace print_utils

#endif //PRINT_UTILS_H

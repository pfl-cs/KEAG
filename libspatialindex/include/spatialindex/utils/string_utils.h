#ifndef STRING_UTILS_H
#define STRING_UTILS_H
#include <iostream>
#include <algorithm>

namespace string_utils
{
    void str_self_tolower(std::string &s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
    }

    std::string str_tolower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
        return s;
    }

    bool is_integer(const std::string& s)
    {
        return !s.empty() && std::find_if(s.begin(),
            s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
    }
} // namespace string_utils
#endif //STRING_UTILS_H

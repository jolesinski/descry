#ifndef DESCRY_TEST_UTILS_H
#define DESCRY_TEST_UTILS_H

#include <algorithm>

namespace descry { namespace test { namespace utils {

template<class T>
inline bool all_zeros(const T &container) {
    return std::all_of(std::begin(container), std::end(container),
                       [](const auto &elem) { return elem == 0; });
}

template<class T>
inline bool all_finite(const T &container) {
    return std::all_of(std::begin(container), std::end(container),
                       [](const auto &elem) { return std::isfinite(elem); });
}

template<class T>
inline bool all_normal(const T &container) {
    return std::all_of(std::begin(container), std::end(container),
                       [](const auto &elem) { return std::isnormal(elem); });
}

}}}

#endif //DESCRY_TEST_UTILS_H

#ifndef DESCRY_CUPCL_UNIQUE_H
#define DESCRY_CUPCL_UNIQUE_H

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace std {

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}

#endif //DESCRY_CUPCL_UNIQUE_H

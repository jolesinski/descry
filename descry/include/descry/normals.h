
#ifndef DESCRY_NORMALS_H
#define DESCRY_NORMALS_H

#include <descry/common.h>
#include <descry/image.h>

namespace descry {

class NormalEstimation {
public:
    bool configure(const Config& config);
    Normals::Ptr compute(const Image& image);
private:
    std::function<Normals::Ptr(const Image&)> _nest;
};

}

#endif //DESCRY_NORMALS_H


#ifndef DESCRY_NORMALS_H
#define DESCRY_NORMALS_H

#include <descry/image.h>

namespace descry {

class NormalEstimation {
public:
    bool configure(const Config& config);
    DualNormals compute(const Image& image);
private:
    std::function<DualNormals(const Image&)> _nest;
};

}

#endif //DESCRY_NORMALS_H

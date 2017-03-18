
#ifndef DESCRY_NORMALS_H
#define DESCRY_NORMALS_H

#include <descry/image.h>
#include <descry/config/normals.h>

namespace descry {

class NormalEstimation {
public:
    bool configure(const Config& config);
    DualNormals compute(const Image& image) const;
private:
    std::function<DualNormals(const Image&)> nest_;
};

}

#endif //DESCRY_NORMALS_H


#ifndef DESCRY_NORMALS_H
#define DESCRY_NORMALS_H

#include <descry/common.h>

namespace descry {

class NormalEstimation {
public:
    bool configure(const Config& config);
    Normals::Ptr compute(const PointCloud::ConstPtr& cloud);
private:
    std::function<Normals::Ptr(const PointCloud::ConstPtr&)> _nest;
};

}

#endif //DESCRY_NORMALS_H

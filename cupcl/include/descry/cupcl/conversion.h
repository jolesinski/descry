#ifndef DESCRY_CUPCL_CONVERSION_H
#define DESCRY_CUPCL_CONVERSION_H

#include <descry/common.h>
#include <descry/cupcl/memory.h>
#include <pcl/point_types.h>

#ifdef CUPCL_MOCK
#include <pcl/common/io.h>
#endif

namespace descry { namespace cupcl {
#ifndef CUPCL_MOCK
DualShapeCloud convertToXYZ(const DualConstFullCloud& points);
#else
inline DualShapeCloud convertToXYZ(const DualConstFullCloud& points) {
    auto shapeCloud = make_cloud<ShapePoint>();
    pcl::copyPointCloud(*points.host(), *shapeCloud);
    return DualShapeCloud(shapeCloud);
}
#endif
}}

#endif //DESCRY_CUPCL_CONVERSION_H

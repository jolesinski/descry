#ifndef DESCRY_CUPCL_NORMALS_H
#define DESCRY_CUPCL_NORMALS_H

#include <descry/common.h>

namespace descry { namespace cupcl {

#ifndef CUPCL_MOCK
DualNormals computeNormals(const DualShapeCloud& points,
                           const DualPerpective& projection,
                           const float radius);
#else
inline DualNormals computeNormals(const DualShapeCloud& /*points*/,
                                  const DualPerpective& /*projection*/,
                                  const float /*radius*/) { return DualNormals{make_cloud<pcl::Normal>()}; }
#endif
}}

#endif //DESCRY_CUPCL_NORMALS_H

#ifndef DESCRY_CUPCL_NORMALS_H
#define DESCRY_CUPCL_NORMALS_H

#include <descry/cupcl/memory.h>
#include <descry/common.h>

namespace descry { namespace cupcl {

DualNormals computeNormals(const DualShapeCloud& points,
                           const DualPerpective& projection,
                           const float radius);

}}

#endif //DESCRY_CUPCL_NORMALS_H

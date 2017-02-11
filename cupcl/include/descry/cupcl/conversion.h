#ifndef DESCRY_CUPCL_CONVERSION_H
#define DESCRY_CUPCL_CONVERSION_H

#include <descry/common.h>
#include <descry/cupcl/memory.h>
#include <pcl/point_types.h>

namespace descry { namespace cupcl {

DualShapeCloud convertToXYZ(const DualConstFullCloud& points);

}}

#endif //DESCRY_CUPCL_CONVERSION_H

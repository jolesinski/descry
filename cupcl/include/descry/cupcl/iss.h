
#ifndef CUPCL_ISS_H
#define CUPCL_ISS_H

#include <descry/common.h>

namespace cupcl
{

CloudPtrT
computeISS(const cupcl::Cloud<PointT>& cloud,
           const float resolution,
           const float eps1,
           const float eps2,
           const float eps3);

}

#endif //CUPCL_ISS_H

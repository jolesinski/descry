
#ifndef CUPCL_ISS_H
#define CUPCL_ISS_H

#include <descry/common.h>

namespace descry { namespace cupcl {

struct ISSConfig {
    unsigned int min_neighs = 5u;
    bool use_resolution = false;
    float resolution = 1.f;
    float salient_rad = 0.03f;
    float non_max_rad = 0.03f;
    float lambda_ratio_21 = 0.75f;
    float lambda_ratio_32 = 0.75f;
    float lambda_threshold_3 = 0.0005f;
};

DualShapeCloud computeISS(const DualShapeCloud& points,
                          const DualPerpective& projection,
                          const ISSConfig& cfg);

} }

#endif //CUPCL_ISS_H

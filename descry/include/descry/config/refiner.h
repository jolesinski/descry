#ifndef DESCRY_CONFIG_REFINER_H
#define DESCRY_CONFIG_REFINER_H

#include <descry/config/common.h>

namespace descry { namespace config { namespace refiner {

static constexpr auto NODE_NAME = "refinement";

// basic icp
static constexpr auto ICP_TYPE = "icp";
static constexpr auto MAX_ITERATIONS = "max-iterations";
static constexpr auto MAX_CORRESPONDENCE_DISTANCE = "max-correspondence-distance";
static constexpr auto TRANSFORMATION_EPSILON = "transformation-epsilon";
static constexpr auto EUCLIDEAN_FITNESS_THRESH = "euclidean-fitness";
static constexpr auto USE_RECIPROCAL = "use-reciprocal";

}}}

#endif //DESCRY_CONFIG_REFINER_H

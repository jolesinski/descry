#ifndef DESCRY_CONFIG_NORMALS_H
#define DESCRY_CONFIG_NORMALS_H

namespace descry { namespace config { namespace normals {

// PCL OMP implementation
static constexpr auto OMP_TYPE = "omp";
// Params:
static constexpr auto SUPPORT_RAD = "support-radius";
static constexpr auto THREADS = "threads";

// PCL Integral Images implementation
static constexpr auto INTEGRAL_IMAGE_TYPE = "integral-image";
// Params:
static constexpr auto INTEGRAL_METHOD = "integral-method";
static constexpr auto SMOOTHING_SIZE = "smoothing-size";
static constexpr auto MAX_DEPTH_CHANGE = "max-depth-change";

// CUPCL implementation
static constexpr auto CUPCL_TYPE = "cupcl";
// Params:
// SUPPORT_RAD, see OMP

}}}

#endif //DESCRY_CONFIG_NORMALS_H

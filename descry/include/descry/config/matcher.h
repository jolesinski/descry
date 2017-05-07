#ifndef DESCRY_CONFIG_MATCHER_H
#define DESCRY_CONFIG_MATCHER_H

namespace descry { namespace config { namespace matcher {

static constexpr auto NODE_NAME = "matching";

// PCL kdtree flann
static constexpr auto KDTREE_FLANN_TYPE = "kdtree-flann";
// Params required:
static constexpr auto NORM_TYPE = "norm-type";
static constexpr auto NORM_HAMMING = "hamming";
static constexpr auto NORM_L2 = "l2";
// Params optional:
static constexpr auto MAX_NEIGHS = "max-neighs";
static constexpr auto MAX_DISTANCE = "max-distance";
static constexpr auto USE_LOWE = "use-lowe";
static constexpr auto LOWE_RATIO = "lowe-ratio";

}}}

#endif //DESCRY_CONFIG_MATCHER_H

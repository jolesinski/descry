#ifndef DESCRY_CONFIG_MATCHER_H
#define DESCRY_CONFIG_MATCHER_H

namespace descry { namespace config { namespace matcher {

static constexpr auto NODE_NAME = "matching";

// PCL kdtree flann
static constexpr auto KDTREE_FLANN_TYPE = "kdtree-flann";
// Params required:
static constexpr auto MAX_DISTANCE = "max-distance";
// Params optional:
static constexpr auto MAX_NEIGHS = "max-neighs";

}}}

#endif //DESCRY_CONFIG_MATCHER_H

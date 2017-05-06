#ifndef DESCRY_CONFIG_DESCRIPTORS_H
#define DESCRY_CONFIG_DESCRIPTORS_H

namespace descry { namespace config { namespace descriptors {

static constexpr auto NODE_NAME = "descriptors";

// SHOT PCL implentation
static constexpr auto SHOT_PCL_TYPE = "shot";
// FPFH PCL implentation
static constexpr auto FPFH_PCL_TYPE = "fpfh";
// Params:
static constexpr auto SUPPORT_RAD = "support-radius";
static constexpr auto THREADS = "threads";

// ORB
static constexpr auto ORB_TYPE = "orb";
// Params:
static constexpr auto MAX_FEATURES = "max-features";
static constexpr auto EDGE_THRESH = "edge-thresh";
static constexpr auto FAST_THRESH = "fast-thresh";
static constexpr auto FIRST_LEVEL = "first-level";
static constexpr auto NUM_LEVELS = "num-levels";
static constexpr auto PATCH_SIZE = "patch-size";
static constexpr auto SCALE_FACTOR = "scale-factor";
static constexpr auto SCORE_TYPE = "score-type";
static constexpr auto WTA_K = "wta-k";

}}}

#endif //DESCRY_CONFIG_DESCRIPTORS_H

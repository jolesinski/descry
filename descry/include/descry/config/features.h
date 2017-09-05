#ifndef DESCRY_CONFIG_FEATURES_H
#define DESCRY_CONFIG_FEATURES_H

#include <descry/config/common.h>

namespace descry { namespace config { namespace features {

static constexpr auto NODE_NAME = "features";

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


// AKAZE
static constexpr auto AKAZE_TYPE = "akaze";
static constexpr auto AKAZE_SIZE = "akaze-size";
static constexpr auto AKAZE_CHANNELS = "akaze-channels";
static constexpr auto AKAZE_THRESH = "akaze-thresh";
static constexpr auto NUM_OCTAVES = "num-octaves";
static constexpr auto OCTAVE_LAYERS = "octave-layers";

// SIFT
static constexpr auto SIFT_TYPE = "sift";
static constexpr auto RETAIN_FEATURES = "retain-features";
//static constexpr auto OCTAVE_LAYERS = "octave-layers";
static constexpr auto CONTRAST_THRESH = "contrast-thresh";
// static constexpr auto EDGE_THRESH = "edge-thresh";
static constexpr auto SIGMA = "sigma";

}}}

#endif //DESCRY_CONFIG_FEATURES_H

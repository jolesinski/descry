#ifndef DESCRY_CONFIG_VERIFIER_H
#define DESCRY_CONFIG_VERIFIER_H

#include <descry/config/common.h>

namespace descry { namespace config { namespace verifier {

static constexpr auto NODE_NAME = "verification";

// Papazov
static constexpr auto PAPAZOV_TYPE = "papazov";
static constexpr auto RESOLUTION = "resolution";
static constexpr auto INLIER_THRESH = "inlier-threshold";
static constexpr auto SUPPORT_THRESH = "support-threshold";
static constexpr auto PENALTY_THRESH = "penalty-threshold";
static constexpr auto CONFLICT_THRESH = "conflict-threshold";

// Global
static constexpr auto GLOBAL_TYPE = "global";
static constexpr auto OCCLUSION_THRESH = "occlusion-threshold"; // only for unorganized scenes?
static constexpr auto REGULARIZER = "regularizer";
static constexpr auto CLUTTER_RADIUS = "clutter-radius";
static constexpr auto CLUTTER_REGULARIZER = "clutter-regularizer";
static constexpr auto DETECT_CLUTTER = "detect-clutter";
static constexpr auto RADIUS_NORMALS = "radius-normals";

}}}

#endif //DESCRY_CONFIG_VERIFIER_H

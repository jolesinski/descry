#ifndef DESCRY_CONFIG_VIEWER_H
#define DESCRY_CONFIG_VIEWER_H

namespace descry { namespace config { namespace viewer {

static constexpr auto NODE_NAME = "viewer";

// keypoints
static constexpr auto SHOW_ONCE = "show-once";
static constexpr auto SHOW_ONLY = "show-only";
static constexpr auto SHOW_2D = "show-2d";
static constexpr auto KEYPOINT_SIZE = "keypoint-size";

// clusters
static constexpr auto SHOW_MATCHES = "show-matches";
static constexpr auto SHOW_CLUSTERS = "show-clusters";
static constexpr auto SHOW_EMPTY = "show-empty";

}}}

#endif //DESCRY_CONFIG_VIEWER_H

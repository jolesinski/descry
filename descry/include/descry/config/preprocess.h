#ifndef DESCRY_CONFIG_PREPROCESS_H
#define DESCRY_CONFIG_PREPROCESS_H

namespace descry { namespace config { namespace preprocess {

static constexpr auto NODE_NAME = "preprocess";
static constexpr auto PASSTHROUGH = "passthrough";

static constexpr auto SMOOTHING = "smoothing";
// with larger sigma-r filter becomes closer to a Gaussian blur
static constexpr auto SIGMA_R = "sigma-r";
// larger features are smoothed out with larger simga-s
static constexpr auto SIGMA_S = "sigma-s";

static constexpr auto SEGMENTATION = "segmentation";
static constexpr auto MIN_INLIERS = "min-inliers";
static constexpr auto ANGULAR_THRESH = "angular-thresh";
static constexpr auto DISTANCE_THRESH = "distance-thresh";
static constexpr auto PROJECT_POINTS = "project-points";

}}}

#endif //DESCRY_CONFIG_PREPROCESS_H

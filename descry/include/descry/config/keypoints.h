#ifndef DESCRY_CONFIG_KEYPOINTS_H
#define DESCRY_CONFIG_KEYPOINTS_H

namespace descry { namespace config { namespace keypoints {

static constexpr auto NODE_NAME = "keypoints";

// Uniform sampling
static constexpr auto UNIFORM_TYPE = "uniform";
// Params:
static constexpr auto SUPPORT_RAD = "support-radius";

// Intrinsic Shape Signatures
// PCL implementation
static constexpr auto ISS_TYPE = "iss";
// CUPCL implementation
static constexpr auto ISS_CUPCL_TYPE = "iss-cupcl";
// Params:
static constexpr auto SALIENT_RAD = "salient-radius";
static constexpr auto NON_MAX_RAD = "non-max-radius";
static constexpr auto BORDER_RAD = "border-radius";
static constexpr auto NORMAL_RAD = "normal-radius";
static constexpr auto LAMBDA_RATIO_21 = "lambda-ratio-21";
static constexpr auto LAMBDA_RATIO_32 = "lambda-ratio-32";
static constexpr auto BOUNDARY_ANGLE = "boundary-angle";
static constexpr auto LAMBDA_THRESHOLD_3 = "lambda-threshold-3";
static constexpr auto MIN_NEIGHBOURS = "min-neighbours";
static constexpr auto THREADS = "threads";

// Harris 3d
static constexpr auto HARRIS_TYPE = "harris";
// Params:
static constexpr auto METHOD_NAME = "method";
static constexpr auto METHOD_HARRIS = "harris";
static constexpr auto METHOD_NOBLE = "noble";
static constexpr auto METHOD_LOWE = "lowe";
static constexpr auto METHOD_TOMASI = "tomasi";
static constexpr auto METHOD_CURVATURE = "curvature";
static constexpr auto HARRIS_THRESHOLD = "threshold";
static constexpr auto USE_REFINE = "use_refine";
static constexpr auto USE_NONMAX = "use_nonmax";

}}}

#endif //DESCRY_CONFIG_KEYPOINTS_H

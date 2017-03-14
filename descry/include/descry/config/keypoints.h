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

}}}

#endif //DESCRY_CONFIG_KEYPOINTS_H

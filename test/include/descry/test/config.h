#ifndef DESCRY_TEST_CONFIG_H
#define DESCRY_TEST_CONFIG_H

/*
 * Defines configs with comparable step types
 */

#include <descry/common.h>
#include <descry/config/keypoints.h>
#include <descry/config/normals.h>
#include <descry/test/data_paths.h>

namespace descry { namespace test {

inline descry::Config loadFullConfig() { return YAML::LoadFile(descry::test::CONFIG_PATH); }

namespace normals {

inline descry::Config loadConfigOmp() {
    auto cfg = descry::Config();
    cfg["type"] = config::normals::OMP_TYPE;
    cfg[config::normals::SUPPORT_RAD] = 0.02;
    cfg[config::normals::THREADS] = 8;
    return cfg;
}

inline descry::Config loadConfigInt() {
    auto cfg = descry::Config();
    cfg["type"] = config::normals::INTEGRAL_IMAGE_TYPE;
    cfg[config::normals::INTEGRAL_METHOD] = "covariance";
    cfg[config::normals::SMOOTHING_SIZE] = 10.0f;
    cfg[config::normals::MAX_DEPTH_CHANGE] = 0.02f;
    return cfg;
}

inline descry::Config loadConfigCupcl() {
    auto cfg = descry::Config();
    cfg["type"] = config::normals::CUPCL_TYPE;
    cfg[config::keypoints::SUPPORT_RAD] = 0.02;
    return cfg;
}

}

namespace keypoints {

inline descry::Config loadConfigUniform() {
    auto cfg = Config();
    cfg["type"] = "uniform";
    cfg[config::keypoints::SUPPORT_RAD] = 0.03f;
    return cfg;
}

inline descry::Config loadConfigISS() {
    auto cfg = Config();
    cfg["type"] = config::keypoints::ISS_TYPE;
    cfg[config::keypoints::SALIENT_RAD] = 0.05;
    cfg[config::keypoints::NON_MAX_RAD] = 0.15;
    //cfg[config::keypoints::BORDER_RAD] = 0.03;
    //cfg[config::keypoints::NORMAL_RAD] = 0.03;
    cfg[config::keypoints::MIN_NEIGHBOURS] = 5;
    cfg[config::keypoints::THREADS] = 4;
    return cfg;
}

inline descry::Config loadConfigCupcl() {
    auto cfg = Config();
    cfg["type"] = config::keypoints::ISS_CUPCL_TYPE;
    cfg[config::keypoints::SALIENT_RAD] = 0.05;
    cfg[config::keypoints::NON_MAX_RAD] = 0.15;
    cfg[config::keypoints::LAMBDA_RATIO_21] = 0.975;
    cfg[config::keypoints::LAMBDA_RATIO_32] = 0.975;
    cfg[config::keypoints::LAMBDA_THRESHOLD_3] = 0.00005;
    cfg[config::keypoints::MIN_NEIGHBOURS] = 5;
    return cfg;
}

}

} }

#endif //DESCRY_TEST_CONFIG_H

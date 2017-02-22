#ifndef DESCRY_TEST_CONFIG_H
#define DESCRY_TEST_CONFIG_H

/*
 * Defines configs with comparable step types
 */

#include <descry/common.h>
#include <descry/test/data_paths.h>

namespace descry { namespace test {

inline descry::Config loadFullConfig() { return YAML::LoadFile(descry::test::CONFIG_PATH); }

namespace normals {

inline descry::Config loadConfigOmp() {
    auto cfg = descry::Config();
    cfg["type"] = "omp";
    cfg["k-support"] = 10;
    cfg["threads"] = 8;
    return cfg;
}

inline descry::Config loadConfigInt() {
    auto cfg = descry::Config();
    cfg["type"] = "int";
    cfg["method"] = "covariance";
    cfg["smoothing"] = 10.0f;
    cfg["max-depth-change"] = 0.02f;
    return cfg;
}

inline descry::Config loadConfigCupcl() {
    auto cfg = descry::Config();
    cfg["type"] = "cupcl";
    cfg["r-support"] = 0.02;
    return cfg;
}

}

namespace keypoints {

inline descry::Config loadConfigUniform() {
    auto cfg = descry::Config();
    cfg["type"] = "uniform";
    cfg["r-support"] = 0.03f;
    return cfg;
}

inline descry::Config loadConfigISS() {
    auto cfg = descry::Config();
    cfg["type"] = "iss";
    cfg["salient-radius"] = 0.05;
    cfg["non-max-radius"] = 0.15;
    //cfg["border-radius"] = 0.03;
    //cfg["normal-radius"] = 0.03;
    cfg["min-neighbours"] = 5;
    cfg["threads"] = 4;
    return cfg;
}

inline descry::Config loadConfigCupcl() {
    auto cfg = descry::Config();
    cfg["type"] = "iss-cupcl";
    cfg["salient-radius"] = 0.05;
    cfg["non-max-radius"] = 0.15;
    cfg["lambda-ratio-21"] = 0.975;
    cfg["lambda-ratio-32"] = 0.975;
    cfg["lambda-threshold-3"] = 0.00005;
    cfg["min-neighbours"] = 5;
    return cfg;
}

}

} }

#endif //DESCRY_TEST_CONFIG_H

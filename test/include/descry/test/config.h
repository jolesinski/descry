#ifndef DESCRY_TEST_CONFIG_H
#define DESCRY_TEST_CONFIG_H

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
}

} }

#endif //DESCRY_TEST_CONFIG_H

#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/describer.h>

TEST_CASE( "Configuring fpfh pcl describer", "[describer]" ) {
    auto est = descry::Describer<pcl::FPFHSignature33>{};
    auto cfg = YAML::Load("");

    REQUIRE(!est.configure(cfg));

    cfg["type"] = descry::config::describer::FPFH_PCL_TYPE;

    REQUIRE(!est.configure(cfg));

    cfg[descry::config::describer::SUPPORT_RAD] = 0.015;

    REQUIRE(est.configure(cfg));

    cfg.reset(descry::test::loadFullConfig());

    REQUIRE(!est.configure(cfg));
    REQUIRE(est.configure(cfg["sparse"]["descriptors"]));
}

TEST_CASE( "Configuring shot pcl describer", "[describer]" ) {
    auto est = descry::Describer<pcl::SHOT352>{};
    auto cfg = YAML::Load("");

    REQUIRE(!est.configure(cfg));

    cfg["type"] = descry::config::describer::SHOT_PCL_TYPE;

    REQUIRE(!est.configure(cfg));

    cfg[descry::config::describer::SUPPORT_RAD] = 0.015;

    REQUIRE(est.configure(cfg));

    cfg.reset(descry::test::loadFullConfig());

    REQUIRE(!est.configure(cfg));
    REQUIRE(!est.configure(cfg["sparse"]["descriptors"]));
}
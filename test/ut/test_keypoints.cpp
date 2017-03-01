#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/keypoints.h>

#include <algorithm>

TEST_CASE( "Configuring uniform keypoints", "[keypoints]" ) {
    auto kdet = descry::ShapeKeypointDetector{};
    auto cfg = YAML::Load("");

    REQUIRE(!kdet.configure(cfg));

    cfg.SetTag("keypoints");
    cfg["type"] = "uniform";

    REQUIRE(!kdet.configure(cfg));

    cfg[descry::config::keypoints::SUPPORT_RAD] = 0.03f;

    REQUIRE(kdet.configure(cfg));
}

TEST_CASE( "Configuring iss keypoints", "[keypoints]" ) {
    auto kdet = descry::ShapeKeypointDetector{};
    auto cfg = YAML::Load("");

    REQUIRE(!kdet.configure(cfg));

    cfg.SetTag("keypoints");
    cfg["type"] = descry::config::keypoints::ISS_TYPE;

    REQUIRE(!kdet.configure(cfg));

    cfg[descry::config::keypoints::SUPPORT_RAD] = 0.03f;

    REQUIRE(!kdet.configure(cfg));

    cfg[descry::config::keypoints::SALIENT_RAD] = 0.03;

    REQUIRE(!kdet.configure(cfg));

    cfg[descry::config::keypoints::NON_MAX_RAD] = 0.03;

    REQUIRE(kdet.configure(cfg));

    cfg[descry::config::keypoints::SALIENT_RAD] = "asdf";

    REQUIRE(!kdet.configure(cfg));
}

TEST_CASE( "Keypoint duetection on real cloud", "[keypoints]") {
    auto kdet = descry::ShapeKeypointDetector{};

    SECTION("Uniform") {
        REQUIRE(kdet.configure(descry::test::keypoints::loadConfigUniform()));
    }
//    SECTION("ISS") {
//        REQUIRE(kdet.configure(descry::test::keypoints::loadConfigISS()));
//    }
    SECTION("CUPCL") {
        REQUIRE(kdet.configure(descry::test::keypoints::loadConfigCupcl()));
    }

    auto image = descry::Image(descry::test::loadSceneCloud());
    auto keypoints = kdet.compute(image);

    REQUIRE(!keypoints.empty());
    INFO(keypoints.size());

    std::all_of(keypoints.host()->begin(), keypoints.host()->end(),
                [](const auto& p){ return pcl::isFinite(p); });
}
#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/keypoints.h>

TEST_CASE( "Configuring uniform keypoints", "[keypoints]" ) {
    auto kdet = descry::ShapeKeypointDetector{};
    auto cfg = YAML::Load("");

    REQUIRE(!kdet.configure(cfg));

    cfg.SetTag("keypoints");
    cfg["type"] = "uniform";

    REQUIRE(!kdet.configure(cfg));

    cfg["r-support"] = 0.03f;

    REQUIRE(kdet.configure(cfg));
}

TEST_CASE( "Configuring iss keypoints", "[keypoints]" ) {
    auto kdet = descry::ShapeKeypointDetector{};
    auto cfg = YAML::Load("");

    REQUIRE(!kdet.configure(cfg));

    cfg.SetTag("keypoints");
    cfg["type"] = "iss";

    REQUIRE(!kdet.configure(cfg));

    cfg["r-support"] = 0.03f;

    REQUIRE(!kdet.configure(cfg));

    cfg["salient-radius"] = 0.03;

    REQUIRE(!kdet.configure(cfg));

    cfg["non-max-radius"] = 0.03;

    REQUIRE(kdet.configure(cfg));

    cfg["salient-radius"] = "asdf";

    REQUIRE(!kdet.configure(cfg));
}

TEST_CASE( "Keypoint duetection on real cloud", "[keypoints]") {
    auto kdet = descry::ShapeKeypointDetector{};

    SECTION("Uniform") {
        REQUIRE(kdet.configure(descry::test::keypoints::loadConfigUniform()));
    }
    SECTION("ISS") {
        REQUIRE(kdet.configure(descry::test::keypoints::loadConfigISS()));
    }

    auto image = descry::Image(descry::test::loadSceneCloud());
    auto keypoints = kdet.compute(image);

    REQUIRE(!keypoints->empty());
    WARN(keypoints->size());
}
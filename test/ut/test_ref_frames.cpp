#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/test/utils.h>
#include <descry/keypoints.h>
#include <descry/normals.h>
#include <descry/ref_frames.h>

TEST_CASE( "Configuring reference frames", "[ref_frames]" ) {
    auto est = descry::RefFramesEstimation{};
    auto cfg = YAML::Load("");

    REQUIRE(!est.configure(cfg));

    cfg["type"] = descry::config::ref_frames::BOARD_TYPE;

    REQUIRE(!est.configure(cfg));

    cfg[descry::config::ref_frames::SUPPORT_RAD] = 0.015;

    REQUIRE(est.configure(cfg));

    cfg.reset(descry::test::loadFullConfig());

    REQUIRE(!est.configure(cfg));
    REQUIRE(est.configure(cfg["recognizer"]["reference-frames"]));
}

TEST_CASE( "Reference frames estimation on cloud", "[ref_frames]") {
    auto nest = descry::NormalEstimation{};
    nest.configure(descry::test::normals::loadConfigCupcl());
    auto kdet = descry::ShapeKeypointDetector{};
    kdet.configure(descry::test::keypoints::loadConfigCupcl());
    auto rfest = descry::RefFramesEstimation{};
    auto cfg = YAML::Load("");
    cfg["type"] = descry::config::ref_frames::BOARD_TYPE;
    cfg[descry::config::ref_frames::SUPPORT_RAD] = 0.015;
    rfest.configure(cfg);

    SECTION("Planar cloud") {
        // TODO: all equal
    }
    SECTION("Real cloud") {
        auto image = descry::Image(descry::test::loadSceneCloud());
        image.setNormals(nest.compute(image));
        image.setShapeKeypoints(kdet.compute(image));

        auto rfs = rfest.compute(image);
        REQUIRE(!rfs.empty());
        REQUIRE(rfs.size() == image.getShapeKeypoints().size());
        WARN(rfs.size());

        REQUIRE(std::all_of(rfs.host()->begin(), rfs.host()->end(),
                    [](const auto& p){ return descry::test::utils::all_normal(p.rf); }));
    }
}
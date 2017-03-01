#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/test/utils.h>
#include <descry/descriptors.h>
#include <descry/keypoints.h>
#include <descry/normals.h>
#include <descry/ref_frames.h>

TEST_CASE( "Configuring fpfh pcl describer", "[describer]" ) {
    auto est = descry::Describer<pcl::FPFHSignature33>{};
    auto cfg = YAML::Load("");

    REQUIRE(!est.configure(cfg));

    cfg["type"] = descry::config::descriptors::FPFH_PCL_TYPE;

    REQUIRE(!est.configure(cfg));

    cfg[descry::config::descriptors::SUPPORT_RAD] = 0.015;

    REQUIRE(est.configure(cfg));

    cfg.reset(descry::test::loadFullConfig());

    REQUIRE(!est.configure(cfg));
    REQUIRE(est.configure(cfg["sparse"]["descriptors"]));
}

TEST_CASE( "Configuring shot pcl describer", "[describer]" ) {
    auto est = descry::Describer<pcl::SHOT352>{};
    auto cfg = YAML::Load("");

    REQUIRE(!est.configure(cfg));

    cfg["type"] = descry::config::descriptors::SHOT_PCL_TYPE;

    REQUIRE(!est.configure(cfg));

    cfg[descry::config::descriptors::SUPPORT_RAD] = 0.015;

    REQUIRE(est.configure(cfg));

    cfg.reset(descry::test::loadFullConfig());

    REQUIRE(!est.configure(cfg));
    REQUIRE(!est.configure(cfg["sparse"]["descriptors"]));
}

TEST_CASE( "Descriptor estimation on planar cloud", "[describer]") {
    auto nest = descry::NormalEstimation{};
    nest.configure(descry::test::normals::loadConfigCupcl());
    auto kdet = descry::ShapeKeypointDetector{};
    kdet.configure(descry::test::keypoints::loadConfigCupcl());
    auto rfest = descry::RefFramesEstimation{};
    rfest.configure(descry::test::ref_frames::loadConfigBOARD());

    SECTION("Planar cloud") {
        // TODO: all descr equal
    }
    SECTION("Real cloud") {
        auto image = descry::Image(descry::test::loadSceneCloud());
        image.setNormals(nest.compute(image));
        image.setShapeKeypoints(kdet.compute(image));
        image.setRefFrames(rfest.compute(image));

        SECTION("SHOT PCL") {
            auto dest = descry::Describer<pcl::SHOT352>{};
            dest.configure(descry::test::descriptors::loadConfigSHOT());
            auto descriptors = dest.compute(image);
            REQUIRE(!descriptors.empty());
            REQUIRE(descriptors.size() == image.getShapeKeypoints().size());
            WARN(descriptors.size());

            REQUIRE(std::all_of(descriptors.host()->begin(), descriptors.host()->end(),
                                [](const auto& p){ return !descry::test::utils::all_zeros(p.descriptor);}));

            REQUIRE(std::all_of(descriptors.host()->begin(), descriptors.host()->end(),
                                [](const auto& p){ return descry::test::utils::all_finite(p.descriptor);}));
        }

        SECTION("FPFH PCL") {
            auto dest = descry::Describer<pcl::FPFHSignature33>{};
            dest.configure(descry::test::descriptors::loadConfigFPFH());
            auto descriptors = dest.compute(image);
            REQUIRE(!descriptors.empty());
            REQUIRE(descriptors.size() == image.getShapeKeypoints().size());
            WARN(descriptors.size());

            REQUIRE(std::all_of(descriptors.host()->begin(), descriptors.host()->end(),
                                [](const auto& p){ return !descry::test::utils::all_zeros(p.histogram);}));

            REQUIRE(std::all_of(descriptors.host()->begin(), descriptors.host()->end(),
                                [](const auto& p){ return descry::test::utils::all_finite(p.histogram);}));
        }
    }
}
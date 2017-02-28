#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
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


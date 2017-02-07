#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/image.h>
#include <descry/cupcl/memory.h>

TEST_CASE( "Create from real point cloud", "[image]" ) {
    auto cloud = descry::test::loadSceneCloud();
    auto image = descry::Image(cloud);

    REQUIRE(image.get<descry::PointCloud::ConstPtr>() == cloud);
}

TEST_CASE( "Create dual container", "[image]" ) {
    auto cloud = descry::test::loadSceneCloud();
    auto dual = descry::cupcl::DualContainer<descry::Point>(cloud);

    REQUIRE(dual.host() == cloud);
    REQUIRE(dual.device() != nullptr);
    REQUIRE_NOTHROW(dual.download());
    REQUIRE(dual.host() != cloud);
    REQUIRE(std::equal(cloud->begin(), cloud->end(), dual.host()->begin(),
                       [](const auto& rhs, const auto& lhs)
                       { return (!pcl::isFinite(lhs) && !pcl::isFinite(rhs)) ||
                                 (std::tie(lhs.x, lhs.y, lhs.z, lhs.rgba) ==
                                  std::tie(rhs.x, rhs.y, rhs.z, rhs.rgba)); }));
    REQUIRE_NOTHROW(dual.reset());
    REQUIRE(dual.device() == nullptr);
    REQUIRE(dual.host() == nullptr);
    REQUIRE_NOTHROW(dual.reset(cloud));
    REQUIRE(dual.device() != nullptr);
    REQUIRE(dual.host() == cloud);
}
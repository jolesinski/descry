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

    dual.device();

    REQUIRE(dual.host() == cloud);
}
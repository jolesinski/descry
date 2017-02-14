#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/image.h>

TEST_CASE( "Create from real point cloud", "[image]" ) {
    auto cloud = descry::test::loadSceneCloud();
    auto image = descry::Image(cloud);

    REQUIRE(image.getFullCloud().host() == cloud);

    auto& shape = image.getShapeCloud();

    REQUIRE(shape.size() == cloud->size());

//    Needed in debug config
//    auto ptr = descry::ShapeCloud::Ptr(new descry::ShapeCloud);
//    ptr.reset(new descry::ShapeCloud);
//    ptr.reset();
//    ptr.reset(new descry::ShapeCloud);
    auto& h_shape = shape.host();

    REQUIRE(h_shape->size() == cloud->size());
}

/*
TEST_CASE( "Create dual container", "[image_disabled]" ) {
    auto cloud = descry::test::loadSceneCloud();
    auto dual = descry::cupcl::DualContainer<descry::FullPoint>(cloud);

    REQUIRE(dual.host() == cloud);
    REQUIRE(dual.size() == cloud->size());
    REQUIRE(dual.device() != nullptr);
    REQUIRE_NOTHROW(dual.download());
    REQUIRE(dual.host() != cloud);
    REQUIRE(std::equal(cloud->begin(), cloud->end(), dual.host()->begin(),
                       [](const auto& rhs, const auto& lhs)
                       { return (!pcl::isFinite(lhs) && !pcl::isFinite(rhs)) ||
                                 (std::tie(lhs.x, lhs.y, lhs.z, lhs.rgba) ==
                                  std::tie(rhs.x, rhs.y, rhs.z, rhs.rgba)); }));
    REQUIRE_NOTHROW(dual.reset());
    REQUIRE(dual.size() == 0);
    REQUIRE(dual.device() == nullptr);
    REQUIRE(dual.host() == nullptr);
    REQUIRE_NOTHROW(dual.reset(cloud));
    REQUIRE(dual.device() != nullptr);
    REQUIRE(dual.host() == cloud);
}
 */
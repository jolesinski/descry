#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/model.h>
#include <descry/willow.h>

#include <algorithm>

TEST_CASE( "Create model from full point cloud", "[model]" ) {
    auto cloud = descry::test::loadModelCloud();

    auto cfg = descry::Config{};
    constexpr auto views_size = 10u;
    cfg[descry::config::projection::SPHERE_RAD] = 1.0f;
    cfg[descry::config::projection::SPIRAL_TURNS] = 3;
    cfg[descry::config::projection::SPIRAL_DIV] = views_size;

    auto projector = descry::SphericalProjector(cfg, descry::test::loadPerspective());
    auto views = projector.generateViews(cloud);

    REQUIRE(!views.empty());
    REQUIRE(views.size() == views_size);

    for (const auto& view : views) {
        const auto& partial = view.image.getFullCloud().host();
        REQUIRE( !partial->empty() );
        REQUIRE( partial->isOrganized() );
        REQUIRE( !std::all_of(partial->begin(), partial->end(), [](const auto& p){return !pcl::isFinite(p);}) );
    }

}

TEST_CASE( "Load willow model step by step", "[model]" ) {
    auto obj_path = std::string(descry::test::WILLOW_PATH) + "/willow_models/object_10";
    auto cloud = descry::test::loadCloudPCD(obj_path + "/3D_model.pcd");

    auto projector = descry::WillowProjector(obj_path + "/views");
    auto views = projector.generateViews(cloud);

    REQUIRE(!views.empty());
    INFO("Views size " << views.size());

    for (const auto& view : views) {
        const auto& partial = view.image.getFullCloud().host();
        REQUIRE( !partial->empty() );
        REQUIRE( partial->isOrganized() );
        REQUIRE( !std::all_of(partial->begin(), partial->end(), [](const auto& p){return !pcl::isFinite(p);}) );
    }
}


TEST_CASE( "Load single model from willow database", "[model]" ) {
    auto willow = descry::WillowDatabase(descry::test::WILLOW_MODELS_PATH);
    auto model = willow.loadModel("object_10");

    REQUIRE(!model.getFullCloud()->empty());

    const auto& views = model.getViews();
    REQUIRE(!views.empty());
    INFO("Views size " << views.size());

    for (const auto& view : views) {
        const auto& partial = view.image.getFullCloud().host();
        REQUIRE( !partial->empty() );
        REQUIRE( partial->isOrganized() );
        REQUIRE( !std::all_of(partial->begin(), partial->end(), [](const auto& p){return !pcl::isFinite(p);}) );
    }
}

TEST_CASE( "Load whole willow database", "[model]" ) {
    auto willow = descry::WillowDatabase(descry::test::WILLOW_MODELS_PATH);
    auto models = willow.loadDatabase();

    REQUIRE(!models.empty());

    const auto& views = models.at(10).getViews();
    REQUIRE(!views.empty());
    INFO("Views size " << views.size());

    for (const auto& view : views) {
        const auto& partial = view.image.getFullCloud().host();
        REQUIRE( !partial->empty() );
        REQUIRE( partial->isOrganized() );
        REQUIRE( !std::all_of(partial->begin(), partial->end(), [](const auto& p){return !pcl::isFinite(p);}) );
    }
}
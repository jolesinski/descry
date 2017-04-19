#include <catch.hpp>

#include <descry/test/config.h>
#include <descry/willow.h>

TEST_CASE( "Test db config", "[willow]" ) {
    auto cfg = descry::test::loadDBConfig();

    REQUIRE(cfg.size() == 2);
    WARN(cfg["models"].size());
    REQUIRE(cfg["models"].IsMap());
    REQUIRE(cfg["models"]["test"].IsMap());
}

TEST_CASE( "Load single model", "[willow]" ) {
    auto willow = descry::WillowDatabase(descry::test::loadDBConfig()["models"]);
    auto model = willow.loadModel("test");

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

TEST_CASE( "Load single test scene", "[willow]" ) {
    auto test_object = std::string("object_10");
    auto test_set = std::string("T_01_willow_dataset");

    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    auto model = willow.loadModel(test_object);

    auto scenes = willow.loadSingleTest(test_set);
    const auto& scene_cloud = scenes.front().first;
    const auto& instances = scenes.front().second.at(test_object);

    REQUIRE(instances.size() == 1);

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
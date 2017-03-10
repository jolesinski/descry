#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/matching.h>

TEST_CASE( "Configuring kdtree flann matcher", "[matcher]" ) {
    auto matcher = descry::Matcher<pcl::SHOT352>{};
    auto cfg = YAML::Load("");

    REQUIRE(!matcher.configure(cfg));

    cfg["type"] = descry::config::matcher::KDTREE_FLANN_TYPE;

    REQUIRE(!matcher.configure(cfg));

    cfg[descry::config::matcher::MAX_NEIGHS] = 1;

    REQUIRE(!matcher.configure(cfg));

    cfg[descry::config::matcher::MAX_DISTANCE] = 0.25f;

    REQUIRE(matcher.configure(cfg));
}

namespace {

pcl::SHOT352 random_point() {
    pcl::SHOT352 shot;
    for (auto& elem : shot.descriptor) {
        elem = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    return shot;
}

}

TEST_CASE( "Find correspondences", "[matcher]" ) {
    auto matcher = descry::Matcher<pcl::SHOT352>{};
    auto cfg = YAML::Load("");
    cfg["type"] = descry::config::matcher::KDTREE_FLANN_TYPE;
    cfg[descry::config::matcher::MAX_NEIGHS] = 1;
    cfg[descry::config::matcher::MAX_DISTANCE] = 0.25f;

    REQUIRE(matcher.configure(cfg));

    auto model_size = 352u;
    auto scene_size = 10*model_size;

    pcl::PointCloud<pcl::SHOT352>::Ptr model_descr(new pcl::PointCloud<pcl::SHOT352>);
    pcl::PointCloud<pcl::SHOT352>::Ptr scene_descr(new pcl::PointCloud<pcl::SHOT352>);

    auto mul = 0.9f;
    pcl::SHOT352 base;
    for (auto& elem : base.descriptor) {
        elem = 0.2f * mul;
        mul *= 0.9f;
    }

    model_descr->push_back(base);
    for (auto idx = 1u; idx < model_size; ++idx) {
        model_descr->push_back(random_point());
    }

    std::vector<descry::Matcher<pcl::SHOT352>::DualDescriptors> model_duals;
    model_duals.emplace_back(model_descr);
    matcher.setModel(model_duals);

    for (auto idx = 0u; idx < scene_size - 1; ++idx) {
        scene_descr->push_back(random_point());
    }
    scene_descr->push_back(base);

    descry::Matcher<pcl::SHOT352>::DualDescriptors scene(scene_descr);
    auto corrs = matcher.match(scene);

    REQUIRE(corrs.size() == 1);
    REQUIRE(corrs[0]->size() == 1);

    scene_descr->push_back(base);
    scene.reset(scene_descr);

    corrs = matcher.match(scene);

    REQUIRE(corrs.size() == 1);
    REQUIRE(corrs[0]->size() == 2);

    model_descr->push_back(base);
    matcher.setModel(model_duals);
    corrs = matcher.match(scene);

    REQUIRE(corrs.size() == 1);
    REQUIRE(corrs[0]->size() == 2);

    cfg[descry::config::matcher::MAX_NEIGHS] = 2;
    REQUIRE(matcher.configure(cfg));

    matcher.setModel(model_duals);
    corrs = matcher.match(scene);

    REQUIRE(corrs.size() == 1);
    REQUIRE(corrs[0]->size() == 4);

    model_duals.emplace_back(model_descr);
    matcher.setModel(model_duals);
    corrs = matcher.match(scene);

    REQUIRE(corrs.size() == 2);
    REQUIRE(corrs[0]->size() == 4);
}
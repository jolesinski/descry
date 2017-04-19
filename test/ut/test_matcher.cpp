#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/matching.h>
#include <descry/preprocess.h>
#include <descry/willow.h>
#include <descry/descriptors.h>

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


SCENARIO( "Find corrs in real cloud", "[matcher]" ) {
    GIVEN("Preprocessed model and scene") {
        auto prep = descry::Preprocess{};
        REQUIRE(prep.configure(descry::test::preprocess::loadConfigCupclBoard()));

        auto image = descry::Image(descry::test::loadSceneCloud());
        prep.process(image);

        auto willow = descry::WillowDatabase(descry::test::loadDBConfig());
        auto model = willow.loadModel("test");
        model.prepare(prep);
        WHEN("Descriptor is SHOT") {
            auto dest = descry::Describer<pcl::SHOT352>{};
            dest.configure(descry::test::descriptors::loadConfigSHOT());
            auto scene_descr = dest.compute(image);

            THEN("Matcher should find valid corrs") {
                auto matcher = descry::Matcher<pcl::SHOT352>{};

                auto max_dist = 0.25f;
                auto cfg = descry::test::matching::loadConfigKdtreeFlann();
                cfg[descry::config::matcher::MAX_DISTANCE] = max_dist;
                matcher.configure(cfg);

                std::vector<descry::Matcher<pcl::SHOT352>::DualDescriptors> model_descr;
                for(const auto& view : model.getViews())
                    model_descr.emplace_back(dest.compute(view.image));

                matcher.setModel(model_descr);

                auto corrs = matcher.match(scene_descr);

                REQUIRE(corrs.size() == model.getViews().size());
                for(const auto& test_corr : *corrs[0]) {
                    auto &model_d = model_descr[0].host()->at(static_cast<unsigned>(test_corr.index_query));
                    auto &scene_d = scene_descr.host()->at(static_cast<unsigned>(test_corr.index_match));

                    auto model_map = Eigen::Map<Eigen::Matrix<float, 352, 1>>(model_d.descriptor);
                    auto scene_map = Eigen::Map<Eigen::Matrix<float, 352, 1>>(scene_d.descriptor);

                    auto distance = (model_map - scene_map).squaredNorm();
                    REQUIRE(distance < max_dist);
                    REQUIRE(distance == Approx(test_corr.distance));
                }
            }
        }
        WHEN("Descriptor is FPFH") {
            auto dest = descry::Describer<pcl::FPFHSignature33>{};
            dest.configure(descry::test::descriptors::loadConfigFPFH());
            auto scene_descr = dest.compute(image);

            THEN("Matcher should find valid corrs") {
                auto matcher = descry::Matcher<pcl::FPFHSignature33>{};

                const auto max_dist = 0.25f;
                auto cfg = descry::test::matching::loadConfigKdtreeFlann();
                cfg[descry::config::matcher::MAX_DISTANCE] = max_dist;
                matcher.configure(cfg);

                std::vector<descry::Matcher<pcl::FPFHSignature33>::DualDescriptors> model_descr;
                for(const auto& view : model.getViews())
                    model_descr.emplace_back(dest.compute(view.image));

                matcher.setModel(model_descr);

                auto corrs = matcher.match(scene_descr);

                REQUIRE(corrs.size() == model.getViews().size());
                for(const auto& test_corr : *corrs[0]) {
                    auto &model_d = model_descr[0].host()->at(static_cast<unsigned>(test_corr.index_query));
                    auto &scene_d = scene_descr.host()->at(static_cast<unsigned>(test_corr.index_match));

                    auto model_map = Eigen::Map<Eigen::Matrix<float, 33, 1>>(model_d.histogram);
                    auto scene_map = Eigen::Map<Eigen::Matrix<float, 33, 1>>(scene_d.histogram);

                    auto distance = (model_map - scene_map).squaredNorm();
                    REQUIRE(distance < max_dist);
                    REQUIRE(distance == Approx(test_corr.distance));
                }
            }
        }
    }
}
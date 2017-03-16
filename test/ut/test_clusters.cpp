#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/clusters.h>
#include <descry/matching.h>
#include <descry/preprocess.h>
#include <descry/willow.h>
#include <descry/descriptors.h>

SCENARIO( "Find clusters in real cloud", "[clusters]" ) {
    GIVEN("Preprocessed model and scene") {
        auto prep = descry::Preprocess{};
        REQUIRE(prep.configure(descry::test::preprocess::loadConfigCupclBoard()));

        auto image = descry::Image(descry::test::loadSceneCloud());
        prep.process(image);

        auto willow = descry::WillowDatabase(descry::test::WILLOW_MODELS_PATH);
        auto model = willow.loadModel("test");
        model.prepare(prep);

        WHEN("Descriptor is FPFH") {
            using Descriptor = pcl::FPFHSignature33;

            auto matcher = descry::Matcher<Descriptor>{};
            matcher.configure(descry::test::matching::loadConfigKdtreeFlann());

            auto dest = descry::Describer<Descriptor>{};
            dest.configure(descry::test::descriptors::loadConfigFPFH());
            auto scene_descr = dest.compute(image);

            std::vector<descry::Matcher<Descriptor>::DualDescriptors> model_descr;
            for(const auto& view : model.getViews())
                model_descr.emplace_back(dest.compute(view.image));
            matcher.setModel(model_descr);

            auto corrs = matcher.match(scene_descr);
            WARN(corrs.size());

            THEN("Should find valid instances") {
                auto clusterer = descry::Clusterizer{};
                clusterer.configure(descry::test::clusters::loadConfigHough());
                clusterer.setModel(model);

                auto instances = clusterer.compute(image, corrs);

                REQUIRE(!instances.poses.empty());
            }
        }
    }
}

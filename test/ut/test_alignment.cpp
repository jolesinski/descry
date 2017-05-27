#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/test/utils.h>
#include <descry/alignment.h>
#include <descry/preprocess.h>
#include <descry/willow.h>

SCENARIO( "Sparse alignment", "[alignment]" ) {
    GIVEN("A preprocessed image and model") {
        auto prep = descry::Preprocess{};
        REQUIRE(prep.configure(descry::test::preprocess::loadConfigCupclBoard()));

        auto image = descry::Image(descry::test::loadSceneCloud());
        prep.process(image);

        auto willow = descry::WillowDatabase(descry::test::loadDBConfig());
        auto model = willow.loadModel("test");
        model.prepare(prep);

        WHEN("Aligner is sparse with shot and hough") {
            auto aligner = descry::Aligner{};
            auto align_cfg = descry::Config{};
            align_cfg["type"] = descry::config::aligner::SPARSE_TYPE;
            align_cfg[descry::config::descriptors::NODE_NAME] = descry::test::descriptors::loadConfigSHOT();
            align_cfg[descry::config::matcher::NODE_NAME] = descry::test::matching::loadConfigKdtreeFlann();
            align_cfg[descry::config::clusters::NODE_NAME] = descry::test::clusters::loadConfigHough();

            REQUIRE(aligner.configure(align_cfg));

            aligner.train(model);

            THEN("At least one instance should be found") {
                auto instances = aligner.compute(image);
                REQUIRE(!instances.poses.empty());
                INFO("Found poses " << instances.poses.size());
                for (const auto& pose : instances.poses)
                {
                    INFO("Pose " << pose);
                    REQUIRE(pose.block(0,0,3,3).determinant() == Approx(1));
                }
            }
        }
    }


}
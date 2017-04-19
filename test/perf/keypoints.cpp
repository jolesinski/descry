#include <benchmark/benchmark.h>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/keypoints.h>
#include <descry/willow.h>

static void BM_Keypoints_Time(benchmark::State& state, descry::Config&& cfg) {
    //TODO: load single model, and single test scene with ground truth
    // calculate model views and scene keypoints, extract gt keys from scene keys
    // calculate how many gt keys has a model match within given precision distance
    // Write class WillowTestSet that inherits from WillowDatabase, filter gt points based on pose and full 3d model

    // load model and prepare
    auto model_cfg = descry::Config{};
    model_cfg[descry::config::keypoints::NODE_NAME] = cfg;

    auto prep = descry::Preprocess{};
    prep.configure(model_cfg);

    auto willow = descry::WillowDatabase(descry::test::loadDBConfig()["models"]);
    auto model = willow.loadModel("object_10");
    model.prepare(prep);

    std::cout << "Model prepared" << std::endl;

    auto image = descry::Image(descry::test::loadSceneCloud());
    auto kdet = descry::ShapeKeypointDetector{};

    if(!kdet.configure(cfg))
        state.SkipWithError("Configuration failed");

    double gt_keys = 1;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(kdet.compute(image));
        gt_keys += 1;
    }

    state.counters["repeatability_rate"] = gt_keys;
}
//BENCHMARK_CAPTURE(BM_Keypoints_Time, uniform, descry::test::keypoints::loadConfigUniform())->Unit(benchmark::kMillisecond);
//BENCHMARK_CAPTURE(BM_Keypoints, iss, descry::test::keypoints::loadConfigISS())->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Keypoints_Time, iss_cupcl, descry::test::keypoints::loadConfigCupcl())->Unit(benchmark::kMillisecond);
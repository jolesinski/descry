#include <benchmark/benchmark.h>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/keypoints.h>

static void BM_Keypoints_Time(benchmark::State& state, descry::Config&& cfg) {
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
BENCHMARK_CAPTURE(BM_Keypoints_Time, uniform, descry::test::keypoints::loadConfigUniform())->Unit(benchmark::kMillisecond);
//BENCHMARK_CAPTURE(BM_Keypoints, iss, descry::test::keypoints::loadConfigISS())->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Keypoints_Time, iss_cupcl, descry::test::keypoints::loadConfigCupcl())->Unit(benchmark::kMillisecond);
#include <benchmark/benchmark.h>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/normals.h>

static void BM_Normals(benchmark::State& state, descry::Config&& cfg) {
    auto image = descry::Image(descry::test::loadSceneCloud());
    auto nest = descry::NormalEstimation{};

    if(!nest.configure(cfg))
        state.SkipWithError("Configuration failed");

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(nest.compute(image));
    }
}
//BENCHMARK_CAPTURE(BM_Normals, omp, descry::test::normals::loadConfigOmp())->Unit(benchmark::kMillisecond);
//BENCHMARK_CAPTURE(BM_Normals, int, descry::test::normals::loadConfigInt())->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Normals, cupcl, descry::test::normals::loadConfigCupcl())->Unit(benchmark::kMillisecond);
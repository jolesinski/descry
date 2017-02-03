#include <nonius/nonius.h++>

#include <descry/test/data.h>
#include <descry/normals.h>

NONIUS_BENCHMARK("normals_omp", [](nonius::chronometer meter){
    auto cloud = descry::test::loadScene();
    auto nest = descry::NormalEstimation{};
    auto cfg = YAML::LoadFile(descry::test::CONFIG_PATH)["sparse"]["normals"];

    if(nest.configure(cfg))
        meter.measure([&nest, &cloud](){ return nest.compute(cloud); });
})
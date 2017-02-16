#include <nonius/nonius.h++>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/normals.h>

NONIUS_BENCHMARK("normals_omp", [](nonius::chronometer meter){
    auto image = descry::Image(descry::test::loadSceneCloud());
    auto nest = descry::NormalEstimation{};

    if(nest.configure(descry::test::normals::loadConfigOmp()))
        meter.measure([&nest, &image](){ return nest.compute(image); });
})

NONIUS_BENCHMARK("normals_int", [](nonius::chronometer meter){
    auto image = descry::Image(descry::test::loadSceneCloud());
    auto nest = descry::NormalEstimation{};

    if(nest.configure(descry::test::normals::loadConfigInt()))
        meter.measure([&nest, &image](){ return nest.compute(image); });
})

NONIUS_BENCHMARK("normals_cupcl", [](nonius::chronometer meter){
    auto image = descry::Image(descry::test::loadSceneCloud());
    auto nest = descry::NormalEstimation{};

    if(nest.configure(descry::test::normals::loadConfigCupcl()))
            meter.measure([&nest, &image](){ return nest.compute(image); });
})
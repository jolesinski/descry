#include <nonius/nonius.h++>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/normals.h>

//TODO: add a macro

NONIUS_BENCHMARK("normals_omp", [](nonius::chronometer meter){
    auto cloud = descry::test::loadScene();
    auto nest = descry::NormalEstimation{};

    if(nest.configure(descry::test::normals::loadConfigOmp()))
        meter.measure([&nest, &cloud](){ return nest.compute(cloud); });
})

NONIUS_BENCHMARK("normals_int", [](nonius::chronometer meter){
    auto cloud = descry::test::loadScene();
    auto nest = descry::NormalEstimation{};

    if(nest.configure(descry::test::normals::loadConfigInt()))
        meter.measure([&nest, &cloud](){ return nest.compute(cloud); });
})
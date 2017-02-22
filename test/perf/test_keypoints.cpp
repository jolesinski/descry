#include <nonius/nonius.h++>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/keypoints.h>

NONIUS_BENCHMARK("keypoints_uniform", [](nonius::chronometer meter){
    auto image = descry::Image(descry::test::loadSceneCloud());
    auto kdet = descry::ShapeKeypointDetector{};

    if(kdet.configure(descry::test::keypoints::loadConfigUniform()))
        meter.measure([&kdet, &image](){ return kdet.compute(image); });
})

//NONIUS_BENCHMARK("keypoints_iss", [](nonius::chronometer meter){
//    auto image = descry::Image(descry::test::loadSceneCloud());
//    auto kdet = descry::ShapeKeypointDetector{};
//
//    if(kdet.configure(descry::test::keypoints::loadConfigISS()))
//            meter.measure([&kdet, &image](){ return kdet.compute(image); });
//})

NONIUS_BENCHMARK("keypoints_iss_cupcl", [](nonius::chronometer meter){
    auto image = descry::Image(descry::test::loadSceneCloud());
    auto kdet = descry::ShapeKeypointDetector{};

    if(kdet.configure(descry::test::keypoints::loadConfigCupcl()))
            meter.measure([&kdet, &image](){ return kdet.compute(image); });
})
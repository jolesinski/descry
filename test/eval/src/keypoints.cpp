#include <algorithm>
#include <iostream>

#include <descry/keypoints.h>
#include <descry/willow.h>
#include <descry/test/config.h>
#include <descry/test/data.h>

#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

void eval(const descry::View& model,
          const descry::ShapeKeypoints& scene,
          const descry::AlignedVector<descry::Pose>& gt) {
    std::cout << "Model keys " << model.image.getShapeKeypoints().size() << " scene " << scene.size() << std::endl;

//    descry::FullCloud::Ptr transformed(new descry::FullCloud);
//    pcl::transformPointCloud (*model.image.getFullCloud().host(), *transformed, gt.front() * model.viewpoint);
//
//    pcl::visualization::CloudViewer viewer("Cloud Viewer");
//    viewer.showCloud(transformed, "model");
//    viewer.showCloud(scene.host(), "scene");
//    while (!viewer.wasStopped ())
//        continue;
}

int main() {
    //TODO: pick best pose view from model or merge all
    // calculate keypoints, extract gt keys from scene keys (FULL CLOUD needed)
    // calculate how many gt keys has a model match within given precision distance

    // setup
    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    constexpr auto test_name = "T_01_willow_dataset";
    constexpr auto model_name = "object_10";

    // load data
    auto test_data = willow.loadSingleTest(test_name);
    auto model = willow.loadModel(model_name);
    auto image = descry::Image(test_data.front().first);

    auto gt_handle = std::find_if(std::begin(test_data.front().second), std::end(test_data.front().second),
                           [](const auto& el){ return el.first == model_name; });
    if (gt_handle == std::end(test_data.front().second))
        return EXIT_FAILURE;

    for (const auto& gt : gt_handle->second)
        std::cout << gt_handle->second.front() << std::endl;

    // prepare
    auto model_cfg = descry::Config{};
    model_cfg[descry::config::keypoints::NODE_NAME] = descry::test::keypoints::loadConfigCupcl();
    auto prep = descry::Preprocess{};
    prep.configure(model_cfg);

    auto kdet = descry::ShapeKeypointDetector{};
    if(!kdet.configure(model_cfg[descry::config::keypoints::NODE_NAME]))
        return EXIT_FAILURE;

    // compute
    model.prepare(prep);
    auto keys = kdet.compute(image);

//    eval(model.getViews().front(), keys, gt_handle->second);
    for (const auto& view : model.getViews()) {
        eval(view, keys, gt_handle->second);
    }

    return EXIT_SUCCESS;
}

#include <descry/viewer.h>
#include <descry/config/aligner.h>
#include <descry/config/clusters.h>
#include <descry/config/common.h>
#include <descry/config/normals.h>
#include <descry/config/keypoints.h>
#include <descry/descriptors.h>
#include <descry/keypoints.h>

#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <descry/model.h>

namespace descry {

namespace {

template <typename... Args>
pcl::visualization::PCLVisualizer make_viewer(Args... args) {
    auto viewer = pcl::visualization::PCLVisualizer{args...};

    viewer.setBackgroundColor(0, 0, 0);
    viewer.addCoordinateSystem(.3);
    viewer.initCameraParameters();

    return viewer;
}

void add_image_with_keypoints(const Image& image, const Keypoints& keys, const double size, const std::string& name,
                              pcl::visualization::PCLVisualizer& vis) {
    auto rgb = pcl::visualization::PointCloudColorHandlerRGBField<FullPoint>{image.getFullCloud().host()};
    vis.addPointCloud(image.getFullCloud().host(), rgb, name);
    vis.addPointCloud(keys.getShape().host(), name+config::keypoints::NODE_NAME);
    vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, name+config::keypoints::NODE_NAME);
    vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, name+config::keypoints::NODE_NAME);
}

}

void Viewer<Normals>::show(const FullCloud::ConstPtr& image, const Normals::ConstPtr& normals) const {
    // scalar only for now
    if (!cfg_.IsScalar())
        return;

    auto viewer = make_viewer(config::normals::NODE_NAME);
    auto rgb = pcl::visualization::PointCloudColorHandlerRGBField<FullPoint>{image};
    viewer.addPointCloud(image, rgb, config::SCENE_NODE);
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, config::SCENE_NODE);
    viewer.addPointCloudNormals<FullPoint, pcl::Normal> (image, normals, 10, 0.05, config::normals::NODE_NAME);

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void Viewer<Keypoints>::show(const Image& image, const Keypoints& keypoints) {
    if (!cfg_.IsMap())
        return;

    ++show_count_;
    auto show_once = cfg_[config::viewer::SHOW_ONCE].as<bool>(false);
    if (show_once && show_count_ > 1)
        return;

    auto show_only = cfg_[config::viewer::SHOW_ONLY].as<std::vector<unsigned int>>(std::vector<unsigned int>{});
    if (!show_only.empty() && std::count(show_only.begin(), show_only.end(), show_count_) == 0)
        return;

    auto viewer = make_viewer(config::keypoints::NODE_NAME);

    auto keypoint_size = cfg_[config::viewer::KEYPOINT_SIZE].as<double>(5.0);
    add_image_with_keypoints(image, keypoints, keypoint_size, config::SCENE_NODE, viewer);

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void Viewer<Clusterizer>::addModel(const Model& model, const std::vector<KeyFrameHandle>& m_keyframes) {
    if (!cfg_.IsMap())
        return;

    std::vector<FullCloud::ConstPtr> views;
    for (const auto& view : model.getViews()) {
        views.emplace_back(view.image.getFullCloud().host());
    }

    show_ = [&, views, m_keyframes](const Image& scene, const KeyFrameHandle& keyframe,
               const std::vector<pcl::CorrespondencesPtr>& corrs){
        if (!cfg_.IsMap())
            return;

        auto viewer = make_viewer(config::clusters::NODE_NAME);

        auto show_once = cfg_[config::viewer::SHOW_ONCE].as<bool>(false);
        auto show_only = cfg_[config::viewer::SHOW_ONLY].as<std::vector<unsigned int>>(std::vector<unsigned int>{});
        auto keypoint_size = cfg_[config::viewer::KEYPOINT_SIZE].as<double>(5.0);
        assert(keyframe.keys != nullptr);

        for (auto idx = 0u; idx < views.size(); ++idx) {
            if (idx > 0 && show_once)
                return;
            if (!show_only.empty() && std::count(show_only.begin(), show_only.end(), idx) == 0)
                continue;

            add_image_with_keypoints(scene, *keyframe.keys, keypoint_size, config::SCENE_NODE, viewer);
            add_image_with_keypoints(views.at(idx), *m_keyframes.at(idx).keys,
                                     keypoint_size, config::MODEL_NODE, viewer);
            viewer.addCorrespondences<ShapePoint>(m_keyframes.at(idx).keys->getShape().host(),
                                                  keyframe.keys->getShape().host(), *corrs.at(idx));

            while (!viewer.wasStopped()) {
                viewer.spinOnce(100);
            }
            viewer.removeAllPointClouds();
            viewer.removeCorrespondences();
            viewer.resetStoppedFlag();
        }

    };

}

void Viewer<Clusterizer>::show(const Image& scene, const KeyFrameHandle& keyframe,
                               const std::vector<pcl::CorrespondencesPtr>& corrs) {
    if (!show_)
        return;

    show_(scene, keyframe, corrs);
}

void Viewer<Aligner>::show(const FullCloud::ConstPtr& scene, const Instances& instances) const {
    if (!cfg_.IsScalar())
        return;

    auto viewer = make_viewer(config::aligner::NODE_NAME);
    viewer.addPointCloud(scene, config::SCENE_NODE);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.33, 0.33, 0.33, config::SCENE_NODE);

    auto bell = [](double x, double center){ return 256 * pow((0.5 + (x - center))*(0.5 - (x - center)), 3); };
    auto set_red = [size{instances.poses.size()}, bell](unsigned int idx){ return bell(static_cast<float>(idx)/size, 0); };
    auto set_green = [size{instances.poses.size()}, bell](unsigned int idx){ return bell(static_cast<float>(idx)/size, 0.5); };
    auto set_blue = [size{instances.poses.size()}, bell](unsigned int idx){ return bell(static_cast<float>(idx)/size, 1); };
    for (auto idx = 0u; idx < instances.poses.size(); ++idx) {
        auto transformed = make_cloud<FullPoint>();
        pcl::transformPointCloud(*instances.cloud, *transformed, instances.poses[idx]);

        auto instance_id = std::string("instance_") + std::to_string(idx);
        viewer.addPointCloud(transformed, instance_id);
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                                set_red(idx), set_green(idx), set_blue(idx), instance_id);
    }

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

}
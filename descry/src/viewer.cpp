#include <descry/viewer.h>
#include <descry/config/aligner.h>
#include <descry/config/common.h>
#include <descry/config/normals.h>
#include <descry/config/keypoints.h>
#include <descry/keypoints.h>

#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace descry {

void Viewer<Normals>::show(const FullCloud::ConstPtr& image, const Normals::ConstPtr& normals) const {
    // scalar only for now
    if (!cfg_.IsScalar())
        return;

    auto viewer = pcl::visualization::PCLVisualizer{config::normals::NODE_NAME};
    auto rgb = pcl::visualization::PointCloudColorHandlerRGBField<FullPoint>{image};
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addPointCloud(image, rgb, config::SCENE_NODE);
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, config::SCENE_NODE);
    viewer.addPointCloudNormals<FullPoint, pcl::Normal> (image, normals, 10, 0.05, config::normals::NODE_NAME);
    viewer.addCoordinateSystem(.3);
    viewer.initCameraParameters();

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void Viewer<Keypoints>::show(const Image& image, const Keypoints& keypoints) const {
    if (!cfg_.IsMap())
        return;

    auto keypoint_size = cfg_[config::viewer::KEYPOINT_SIZE].as<double>(5.0);
    std::cout << " keypoint size " << keypoint_size << std::endl;
    auto viewer = pcl::visualization::PCLVisualizer{config::keypoints::NODE_NAME};
    auto rgb = pcl::visualization::PointCloudColorHandlerRGBField<FullPoint>{image.getFullCloud().host()};
    viewer.setBackgroundColor(0, 0, 0);
    viewer.addPointCloud(image.getFullCloud().host(), rgb, config::SCENE_NODE);
    viewer.addPointCloud(keypoints.getShape().host(), config::keypoints::NODE_NAME);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, config::keypoints::NODE_NAME);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                            keypoint_size, config::keypoints::NODE_NAME);
    viewer.addCoordinateSystem(.3);
    viewer.initCameraParameters();

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void Viewer<Aligner>::show(const FullCloud::ConstPtr& scene, const Instances& instances) const {
    if (!cfg_.IsScalar())
        return;

    auto viewer = pcl::visualization::PCLVisualizer{config::aligner::NODE_NAME};
    viewer.setBackgroundColor(0, 0, 0);
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

    viewer.addCoordinateSystem(.3);
    viewer.initCameraParameters();

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }

    viewer.close();
}

}
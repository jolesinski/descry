#include <descry/viewer.h>

#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace descry {

void Viewer<Aligner>::configure(const Config& cfg) {
    if (cfg[config::viewer::NODE_NAME])
        cfg_ = cfg[config::viewer::NODE_NAME];
}

void Viewer<Aligner>::show(const FullCloud::ConstPtr& scene, const Instances& instances) {
    if (!cfg_.IsScalar())
        return;

    auto viewer = pcl::visualization::PCLVisualizer{"Alignment"};
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
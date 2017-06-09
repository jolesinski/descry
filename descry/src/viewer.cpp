#include <descry/viewer.h>
#include <descry/config/aligner.h>
#include <descry/config/clusters.h>
#include <descry/config/common.h>
#include <descry/config/normals.h>
#include <descry/config/keypoints.h>
#include <descry/descriptors.h>
#include <descry/keypoints.h>
#include <descry/model.h>

#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

namespace descry {

namespace {

template <typename... Args>
pcl::visualization::PCLVisualizer make_viewer(Args&&... args) {
    auto viewer = pcl::visualization::PCLVisualizer{std::forward<Args>(args)...};

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

void show_keypoints_2d(const Image& image, const Keypoints& keypoints) {
    auto frame = image.getColorMat();
    auto& color_keys = keypoints.getColor();

    cv::drawKeypoints(frame, color_keys, frame, cv::Scalar(0,0,128), cv::DrawMatchesFlags::DEFAULT);

    cv::namedWindow( config::keypoints::NODE_NAME, cv::WINDOW_AUTOSIZE );
    cv::imshow( config::keypoints::NODE_NAME, frame );
    cv::waitKey();
}

void show_keypoints_3d(const Image& image, const Keypoints& keypoints, double keypoint_size) {
    auto viewer = make_viewer(config::keypoints::NODE_NAME);
    add_image_with_keypoints(image, keypoints, keypoint_size, config::SCENE_NODE, viewer);

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void show_matches_3d(const std::vector<FullCloud::ConstPtr>& views,
                     const std::vector<KeyFrameHandle>& m_keyframes,
                     const Image& scene, const KeyFrameHandle& keyframe,
                     const std::vector<pcl::CorrespondencesPtr>& corrs,
                     const Config& config) {
    auto viewer = make_viewer(config::clusters::NODE_NAME);

    auto show_once = config[config::viewer::SHOW_ONCE].as<bool>(false);
    auto show_only = config[config::viewer::SHOW_ONLY].as<std::vector<unsigned int>>(std::vector<unsigned int>{});
    auto keypoint_size = config[config::viewer::KEYPOINT_SIZE].as<double>(5.0);
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
}

std::vector<cv::DMatch> convert(const pcl::Correspondences& pcl_matches) {
    std::vector<cv::DMatch> matches;

    for (auto match : pcl_matches)
        matches.emplace_back(match.index_match, match.index_query, match.distance);

    return matches;
}

void show_matches_2d(const std::vector<FullCloud::ConstPtr>& views,
                     const std::vector<KeyFrameHandle>& m_keyframes,
                     const Image& scene, const KeyFrameHandle& keyframe,
                     const std::vector<pcl::CorrespondencesPtr>& corrs,
                     const Config& config) {
    auto show_once = config[config::viewer::SHOW_ONCE].as<bool>(false);
    auto show_only = config[config::viewer::SHOW_ONLY].as<std::vector<unsigned int>>(std::vector<unsigned int>{});
    assert(keyframe.keys != nullptr);
    for (auto idx = 0u; idx < views.size(); ++idx) {
        if (idx > 0 && show_once)
            return;
        if (!show_only.empty() && std::count(show_only.begin(), show_only.end(), idx) == 0)
            continue;

        auto scene_frame = scene.getColorMat();
        auto& scene_keys = keyframe.keys->getColor();
        auto view_image = Image{views.at(idx)};
        auto& view_frame = view_image.getColorMat();
        auto& view_keys = m_keyframes.at(idx).keys->getColor();

        cv::Mat display_frame;
        cv::drawMatches(scene_frame, scene_keys, view_frame, view_keys, convert(*corrs.at(idx)), display_frame);

        cv::namedWindow( config::clusters::NODE_NAME, cv::WINDOW_AUTOSIZE );
        cv::imshow( config::clusters::NODE_NAME, display_frame );
        cv::waitKey();
    }
}

void show_clusters_3d(const FullCloud::ConstPtr& view, const KeyFrameHandle& m_keyframe,
                      const Image& scene, const KeyFrameHandle& keyframe,
                      const std::vector<pcl::Correspondences>& clusters) {
    assert(m_keyframe.keys != nullptr);
    assert(keyframe.keys != nullptr);

    auto viewer = make_viewer(config::clusters::NODE_NAME);

    auto count = 0u;
    add_image_with_keypoints(scene, *keyframe.keys, 5.0, config::SCENE_NODE, viewer);
    add_image_with_keypoints(view, *m_keyframe.keys, 5.0, config::MODEL_NODE, viewer);
    for (auto cluster : clusters)
        viewer.addCorrespondences<ShapePoint>(m_keyframe.keys->getShape().host(),
                                              keyframe.keys->getShape().host(), cluster, std::to_string(count++));

    while (!viewer.wasStopped())
        viewer.spinOnce(100);
}

void show_clusters_2d(const FullCloud::ConstPtr& view, const KeyFrameHandle& m_keyframe,
                      const Image& scene, const KeyFrameHandle& keyframe,
                      const std::vector<pcl::Correspondences>& clusters) {
    assert(m_keyframe.keys != nullptr);
    assert(keyframe.keys != nullptr);

    auto scene_frame = scene.getColorMat();
    auto& scene_keys = keyframe.keys->getColor();
    auto view_image = Image{view};
    auto& view_frame = view_image.getColorMat();
    auto& view_keys = m_keyframe.keys->getColor();

    cv::Mat display_frame;
    auto clusters_2d = std::vector<std::vector<cv::DMatch>>{};
    for (auto cluster : clusters)
        clusters_2d.emplace_back(convert(cluster));

    cv::drawMatches(scene_frame, scene_keys, view_frame, view_keys, clusters_2d, display_frame);
    cv::namedWindow( config::clusters::NODE_NAME, cv::WINDOW_AUTOSIZE );
    cv::imshow( config::clusters::NODE_NAME, display_frame );
    cv::waitKey();
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

    auto keypoint_size = cfg_[config::viewer::KEYPOINT_SIZE].as<double>(5.0);

    auto show_2d = cfg_[config::viewer::SHOW_2D].as<bool>(false);
    if (show_2d)
        show_keypoints_2d(image, keypoints);
    else
        show_keypoints_3d(image, keypoints, keypoint_size);
}

void Viewer<Clusterizer>::addModel(const Model& model, const std::vector<KeyFrameHandle>& m_keyframes) {
    if (!cfg_.IsMap())
        return;

    std::vector<FullCloud::ConstPtr> views;
    for (const auto& view : model.getViews()) {
        views.emplace_back(view.image.getFullCloud().host());
    }

    show_matches_ = [&, views, m_keyframes](const Image& scene, const KeyFrameHandle& keyframe,
               const std::vector<pcl::CorrespondencesPtr>& corrs){
        if (!cfg_.IsMap())
            return;

        auto show_2d = cfg_[config::viewer::SHOW_2D].as<bool>(false);
        if (show_2d)
            show_matches_2d(views, m_keyframes, scene, keyframe, corrs, cfg_);
        else
            show_matches_3d(views, m_keyframes, scene, keyframe, corrs, cfg_);
    };

    show_clusters_ = [&, views, m_keyframes](const Image& scene, const KeyFrameHandle& keyframe,
                                        const std::vector<pcl::Correspondences>& clusters, unsigned int idx){
        if (!cfg_.IsMap())
            return;

        auto show_empty = cfg_[config::viewer::SHOW_EMPTY].as<bool>(false);
        if (clusters.empty() && !show_empty)
            return;

        auto show_once = cfg_[config::viewer::SHOW_ONCE].as<bool>(false);
        if (idx > 0 && show_once)
            return;

        auto show_only = cfg_[config::viewer::SHOW_ONLY].as<std::vector<unsigned int>>(std::vector<unsigned int>{});
        if (!show_only.empty() && std::count(show_only.begin(), show_only.end(), idx) == 0)
            return;

        auto show_2d = cfg_[config::viewer::SHOW_2D].as<bool>(false);
        if (show_2d)
            show_clusters_2d(views.at(idx), m_keyframes.at(idx), scene, keyframe, clusters);
        else
            show_clusters_3d(views.at(idx), m_keyframes.at(idx), scene, keyframe, clusters);
    };

}

void Viewer<Clusterizer>::show(const Image& scene, const KeyFrameHandle& keyframe,
                               const std::vector<pcl::CorrespondencesPtr>& corrs) {
    if (!show_matches_ || !cfg_[config::viewer::SHOW_MATCHES].as<bool>(false))
        return;

    show_matches_(scene, keyframe, corrs);
}

void Viewer<Clusterizer>::show(const Image& scene, const KeyFrameHandle& keyframe,
                               const std::vector<pcl::Correspondences>& clustered, unsigned int idx) {
    if (!show_clusters_ || !cfg_[config::viewer::SHOW_CLUSTERS].as<bool>(false))
        return;

    show_clusters_(scene, keyframe, clustered, idx);
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
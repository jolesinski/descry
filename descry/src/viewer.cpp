#include <descry/viewer.h>
#include <descry/descriptors.h>

#ifdef USE_VISUALIZER
#include <descry/config/aligner.h>
#include <descry/config/clusters.h>
#include <descry/config/common.h>
#include <descry/config/normals.h>
#include <descry/config/keypoints.h>
#include <descry/keypoints.h>
#include <descry/matching.h>
#include <descry/model.h>
#include <descry/segmentation.h>

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
    viewer.setCameraPosition(0,-.1,0,0,0,0);

    return viewer;
}

void add_image_with_keypoints(const Image& image, const Keypoints& keys, const double size, const std::string& name,
                              pcl::visualization::PCLVisualizer& vis) {
    auto rgb = pcl::visualization::PointCloudColorHandlerRGBField<FullPoint>{image.getFullCloud().host()};
    vis.addPointCloud(image.getFullCloud().host(), rgb, name);
    if (!keys.empty()) {
        keys.getShape().host()->sensor_origin_ = image.getFullCloud().host()->sensor_origin_;
        keys.getShape().host()->sensor_orientation_ = image.getFullCloud().host()->sensor_orientation_;
        vis.addPointCloud(keys.getShape().host(), name+config::keypoints::NODE_NAME);
        vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, name+config::keypoints::NODE_NAME);
        vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, size, name+config::keypoints::NODE_NAME);
    }
}

template <typename Point>
void add_indexed_color_cloud(const typename pcl::PointCloud<Point>::ConstPtr& cloud, std::string basename,
                             unsigned int idx, unsigned long max, pcl::visualization::PCLVisualizer& vis) {

    static auto bell = [](unsigned int idx, unsigned long max, double center)
                       { return 256 * pow((0.5 + (static_cast<float>(idx)/max - center))*
                                                  (0.5 - (static_cast<float>(idx)/max - center)), 3); };
    auto red = bell(idx, max, 0);
    auto green = bell(idx, max, 0.5);
    auto blue = bell(idx, max, 1);

    auto name_id = basename + std::to_string(idx);
    vis.addPointCloud(cloud, name_id);
    vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                         red, green, blue, name_id);
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
                     const Image& scene, const ModelSceneMatches& matches,
                     const Config& config) {
    auto viewer = make_viewer(config::clusters::NODE_NAME);

    auto show_once = config[config::viewer::SHOW_ONCE].as<bool>(false);
    auto show_only = config[config::viewer::SHOW_ONLY].as<std::vector<unsigned int>>(std::vector<unsigned int>{});
    auto keypoint_size = config[config::viewer::KEYPOINT_SIZE].as<double>(5.0);
    for (auto idx = 0u; idx < views.size(); ++idx) {
        assert(matches.view_corrs.at(idx).view != nullptr);
        if (idx > 0 && show_once)
            return;
        if (!show_only.empty() && std::count(show_only.begin(), show_only.end(), idx) == 0)
            continue;

        add_image_with_keypoints(scene, matches.scene->keypoints, keypoint_size, config::SCENE_NODE, viewer);
        add_image_with_keypoints(Image{views.at(idx)}, matches.view_corrs.at(idx).view->keypoints,
                                 keypoint_size, config::MODEL_NODE, viewer);
        viewer.addCorrespondences<ShapePoint>(matches.view_corrs.at(idx).view->keypoints.getShape().host(),
                                              matches.scene->keypoints.getShape().host(), *matches.view_corrs.at(idx).corrs);

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
                     const Image& scene, const ModelSceneMatches& matches,
                     const Config& config) {
    auto show_once = config[config::viewer::SHOW_ONCE].as<bool>(false);
    auto show_only = config[config::viewer::SHOW_ONLY].as<std::vector<unsigned int>>(std::vector<unsigned int>{});
    for (auto idx = 0u; idx < views.size(); ++idx) {
        assert(matches.view_corrs.at(idx).view != nullptr);

        if (idx > 0 && show_once)
            return;
        if (!show_only.empty() && std::count(show_only.begin(), show_only.end(), idx) == 0)
            continue;

        auto scene_frame = scene.getColorMat();
        auto& scene_keys = matches.scene->keypoints.getColor();
        auto view_image = Image{views.at(idx)};
        auto& view_frame = view_image.getColorMat();
        auto& view_keys = matches.view_corrs.at(idx).view->keypoints.getColor();

        cv::Mat display_frame;
        cv::drawMatches(scene_frame, scene_keys, view_frame, view_keys,
                        convert(*matches.view_corrs.at(idx).corrs), display_frame);

        cv::namedWindow( config::clusters::NODE_NAME, cv::WINDOW_AUTOSIZE );
        cv::imshow( config::clusters::NODE_NAME, display_frame );
        cv::waitKey();
    }
}

void show_clusters_3d(const FullCloud::ConstPtr& view, const KeyFrame& m_keyframe,
                      const Image& scene, const KeyFrame& keyframe,
                      const std::vector<pcl::Correspondences>& clusters) {
    auto viewer = make_viewer(config::clusters::NODE_NAME);

    auto count = 0u;
    add_image_with_keypoints(scene, keyframe.keypoints, 5.0, config::SCENE_NODE, viewer);
    add_image_with_keypoints(Image{view}, m_keyframe.keypoints, 5.0, config::MODEL_NODE, viewer);
    for (auto cluster : clusters)
        viewer.addCorrespondences<ShapePoint>(m_keyframe.keypoints.getShape().host(),
                                              keyframe.keypoints.getShape().host(), cluster, std::to_string(count++));

    while (!viewer.wasStopped())
        viewer.spinOnce(100);
}

void show_clusters_2d(const FullCloud::ConstPtr& view, const KeyFrame& m_keyframe,
                      const Image& scene, const KeyFrame& keyframe,
                      const std::vector<pcl::Correspondences>& clusters) {
    auto scene_frame = scene.getColorMat();
    auto& scene_keys = keyframe.keypoints.getColor();
    auto view_image = Image{view};
    auto& view_frame = view_image.getColorMat();
    auto& view_keys = m_keyframe.keypoints.getColor();

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

void Viewer<Image>::show(const Image& image) const {
    // scalar only for now
    if (!cfg_.IsMap() || !cfg_["enabled"].as<bool>(false))
        return;

    auto show_2d = cfg_[config::viewer::SHOW_2D].as<bool>(false);
    if (show_2d) {
        auto frame = image.getColorMat();
        cv::namedWindow( config::keypoints::NODE_NAME, cv::WINDOW_AUTOSIZE );
        cv::imshow( config::keypoints::NODE_NAME, frame );
        cv::waitKey();
    } else {
        auto viewer = make_viewer("viewer");
        auto rgb = pcl::visualization::PointCloudColorHandlerRGBField<FullPoint>{image.getFullCloud().host()};
        viewer.addPointCloud(image.getFullCloud().host(), rgb, config::SCENE_NODE);
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, config::SCENE_NODE);

        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    }
}

void Viewer<Normals>::show(const FullCloud::ConstPtr& image, const Normals::ConstPtr& normals) const {
    // scalar only for now
    if (!cfg_.IsMap() || !cfg_["enabled"].as<bool>(true))
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
    if (!cfg_.IsMap() || !cfg_["enabled"].as<bool>(true))
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

void Viewer<Clusterizer>::addModel(const Model& model, const std::vector<KeyFrame::Ptr>& m_keyframes) {
    if (!cfg_.IsMap() || !cfg_["enabled"].as<bool>(true))
        return;

    std::vector<FullCloud::ConstPtr> views;
    for (const auto& view : model.getViews()) {
        views.emplace_back(view.image.getFullCloud().host());
    }

    show_matches_ = [&, views, m_keyframes](const Image& scene, const ModelSceneMatches& matches){
        if (!cfg_.IsMap())
            return;

        auto show_2d = cfg_[config::viewer::SHOW_2D].as<bool>(false);
        if (show_2d)
            show_matches_2d(views, scene, matches, cfg_);
        else
            show_matches_3d(views, scene, matches, cfg_);
    };

    show_clusters_ = [&, views, m_keyframes](const Image& scene, const KeyFrame& keyframe,
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
            show_clusters_2d(views.at(idx), *m_keyframes.at(idx), scene, keyframe, clusters);
        else
            show_clusters_3d(views.at(idx), *m_keyframes.at(idx), scene, keyframe, clusters);
    };

}

void Viewer<Clusterizer>::show(const Image& scene, const ModelSceneMatches& matches) {
    if (!show_matches_ || !cfg_[config::viewer::SHOW_MATCHES].as<bool>(false))
        return;

    show_matches_(scene, matches);
}

void Viewer<Clusterizer>::show(const Image& scene, const KeyFrame& keyframe,
                               const std::vector<pcl::Correspondences>& clustered, unsigned int idx) {
    if (!show_clusters_ || !cfg_[config::viewer::SHOW_CLUSTERS].as<bool>(false))
        return;

    show_clusters_(scene, keyframe, clustered, idx);
}

void Viewer<Segmenter>::show(const Image& image, const std::vector<pcl::IndicesPtr>& segments) const {
    if (!cfg_.IsScalar() || !cfg_.as<bool>())
        return;

    auto viewer = make_viewer(config::segments::NODE_NAME);
    viewer.addPointCloud(image.getFullCloud().host(), config::SCENE_NODE);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.33, 0.33, 0.33, config::SCENE_NODE);

    for (auto idx = 0u; idx < segments.size(); ++idx) {
        auto segment_cloud = make_cloud<FullPoint>(*image.getFullCloud().host(), *segments[idx]);
        add_indexed_color_cloud<FullPoint>(segment_cloud, "segment", idx, segments.size(), viewer);
    }

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void Viewer<Aligner>::show(const FullCloud::ConstPtr& scene, const Instances& instances) const {
    if (!cfg_.IsScalar() || !cfg_.as<bool>())
        return;

    auto viewer = make_viewer(config::aligner::NODE_NAME);
    viewer.addPointCloud(scene, config::SCENE_NODE);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.33, 0.33, 0.33, config::SCENE_NODE);

    for (auto idx = 0u; idx < instances.poses.size(); ++idx) {
        auto transformed = make_cloud<FullPoint>();
        pcl::transformPointCloud(*instances.cloud, *transformed, instances.poses[idx]);
        add_indexed_color_cloud<FullPoint>(transformed, "instance_", idx, instances.poses.size(), viewer);
    }

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}
}

#else

namespace descry {
void Viewer<Image>::show(const Image& /*image*/) const {}

void Viewer<Normals>::show(const FullCloud::ConstPtr& /*image*/, const Normals::ConstPtr& /*normals*/) const {}

void Viewer<Keypoints>::show(const Image& /*image*/, const Keypoints& /*keypoints*/) {}

void Viewer<Clusterizer>::addModel(const Model& /*model*/, const std::vector<KeyFrame::Ptr>& /*m_keyframes*/) {}

void Viewer<Clusterizer>::show(const Image& /*scene*/, const ModelSceneMatches& /*matches*/) {}

void Viewer<Clusterizer>::show(const Image& /*scene*/, const KeyFrame& /*keyframe*/,
                               const std::vector<pcl::Correspondences>& /*clustered*/, unsigned int /*idx*/) {}

void Viewer<Segmenter>::show(const Image& /*image*/, const std::vector<pcl::IndicesPtr>& /*segments*/) const {}

void Viewer<Aligner>::show(const FullCloud::ConstPtr& /*scene*/, const Instances& /*instances*/) const {}
}

#endif
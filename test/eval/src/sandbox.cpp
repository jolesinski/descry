#include <iostream>

#include <descry/clusters.h>
#include <descry/descriptors.h>
#include <descry/matching.h>
#include <descry/normals.h>
#include <descry/willow.h>
#include <descry/test/config.h>
#include <descry/test/data.h>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

void view_keys(const descry::Image& image, const descry::Keypoints& keys, std::string window_name) {
    //copy to opencv mat
    auto frame = image.getColorMat();
    auto& color_keys = keys.getColor();

    std::cout << "Detected " << color_keys.size() << std::endl;

    cv::drawKeypoints(frame, color_keys, frame, cv::Scalar(0,0,128), cv::DrawMatchesFlags::DEFAULT);

    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    cv::imshow( window_name, frame );
}

void view_keys(const descry::Model& model, const descry::Keypoints& keys, std::string window_name) {
    //auto keys = compute_harris_3d(image);
    //auto keys = compute_iss_3d(image);

    //copy to opencv mat
    //for (const auto& view : model.getViews()) {
    {
        const auto& view = model.getViews().at(5);

        view_keys(view.image, keys, "model");
        cv::waitKey();
    }
}

std::vector<cv::Point2f> get_corners(const descry::View& view) {
    std::vector<cv::Point2f> corners(4);
    const auto& cloud = view.image.getFullCloud().host();

    int min_i = cloud->width;
    int min_j = cloud->height;
    int max_i = 0;
    int max_j = 0;

    for (int idx : view.mask) {
        int j = idx / cloud->width;
        int i = idx % cloud->width;

        max_i = std::max(max_i, i);
        max_j = std::max(max_j, j);
        min_i = std::min(min_i, i);
        min_j = std::min(min_j, j);
    }

    return {cv::Point2f(min_i, min_j),
            cv::Point2f(min_i, max_j),
            cv::Point2f(max_i, max_j),
            cv::Point2f(max_i, min_j)};
};

void drawViewWithCorners(const descry::View& view) {
    auto corners = get_corners(view);
    cv::Mat frame = view.image.getColorMat().clone();

    cv::line( frame, corners[0], corners[1], cv::Scalar(0, 255, 0), 4 );
    cv::line( frame, corners[1], corners[2], cv::Scalar( 0, 255, 0), 4 );
    cv::line( frame, corners[2], corners[3], cv::Scalar( 0, 255, 0), 4 );
    cv::line( frame, corners[3], corners[0], cv::Scalar( 0, 255, 0), 4 );


    cv::namedWindow( "", cv::WINDOW_AUTOSIZE );
    cv::imshow( "", frame );
    cv::waitKey();
}

std::vector<cv::DMatch> convert(const pcl::Correspondences& pcl_matches) {
    std::vector<cv::DMatch> matches;

    for (auto match : pcl_matches)
        matches.emplace_back(match.index_match, match.index_query, match.distance);

    return matches;
}

std::vector<cv::DMatch> filter(descry::Description<cv::Mat>& scene_d, descry::Description<cv::Mat>& model_d, std::vector<cv::DMatch>& matches, cv::Mat& H, double thresh) {
    if (matches.size() < 4) {
        std::cout << "not enough matches" << std::endl;
        return {};
    }

    std::vector<cv::Point2f> scene;
    std::vector<cv::Point2f> obj;

    for( int i = 0; i < matches.size(); i++ )
    {
        scene.push_back( scene_d.getKeypoints().getColor()[ matches[i].queryIdx ].pt );
        obj.push_back( model_d.getKeypoints().getColor()[ matches[i].trainIdx ].pt );
    }

    cv::Mat inlier_mask;
    std::vector<cv::DMatch> inlier_matches;

    H = findHomography(obj, scene, cv::RANSAC, thresh, inlier_mask);

    if (H.empty()) {
        std::cout << "homography not found" << std::endl;
        return {};
    }

    for(unsigned i = 0; i < matches.size(); i++) {
        if(inlier_mask.at<uchar>(i))
            inlier_matches.emplace_back(matches[i]);
    }

    return inlier_matches;
}

std::vector<cv::DMatch> filter(descry::Description<cv::Mat>& scene_d, descry::Description<cv::Mat>& model_d, pcl::CorrespondencesPtr& pcl_matches, cv::Mat& H, double thresh) {
    auto matches = convert(*pcl_matches);
    return filter(scene_d, model_d, matches, H, thresh);
}

std::vector<cv::DMatch> match(descry::Description<cv::Mat>& scene_d, descry::Description<cv::Mat>& model_d, cv::Mat& H) {
    const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
    const double ransac_thresh = 10.5f; // RANSAC inlier threshold

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector< std::vector<cv::DMatch> > nn_matches;
    matcher.knnMatch(scene_d.getFeatures(), model_d.getFeatures(), nn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        cv::DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2)
            good_matches.emplace_back(first);
    }

    return filter(scene_d, model_d, good_matches, H, ransac_thresh);
}

void view_projection(const descry::Image& scene, const descry::Model& model, const descry::Pose& pose) {
    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
    viewer.setBackgroundColor (0, 0, 0);
    auto transformed = descry::make_cloud<descry::FullPoint>();
    pcl::transformPointCloud(*model.getFullCloud(), *transformed, pose);

    viewer.addPointCloud(scene.getFullCloud().host(), "scene");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "scene");

    viewer.addPointCloud(transformed, "model");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "model");

    viewer.addCoordinateSystem (.3);
    viewer.initCameraParameters ();

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }
}

void recognize(const descry::Config& cfg) {
    auto scene_cfg = cfg["scene"];
    auto model_cfg = cfg["model"];
    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    auto test_name = scene_cfg["name"].as<std::string>();
    auto model_name = model_cfg["name"].as<std::string>();
    auto test_data = willow.loadSingleTest(test_name, 1);
    auto image = descry::Image(test_data.front().first);

    if (scene_cfg[descry::config::normals::NODE_NAME]) {
        auto nest = descry::NormalEstimation{};
        nest.configure(scene_cfg[descry::config::normals::NODE_NAME]);
        image.setNormals(nest.compute(image));
    }

    auto describer = descry::Describer<cv::Mat>{};
    describer.configure(scene_cfg[descry::config::descriptors::NODE_NAME]);
    auto scene_d = describer.compute(image);

//    view_keys(image, "scene");

    // model
    auto model = willow.loadModel(model_name);
    auto& view = model.getViews().at(5);
    //drawViewWithCorners(view);
    describer.configure(model_cfg[descry::config::descriptors::NODE_NAME]);

    auto views_description = std::vector<descry::Description<cv::Mat>>();
    views_description.emplace_back(describer.compute(view.image));
    auto& model_d = views_description.front();
//    view_keys(model, "model");

    std::cout << model_d.getFeatures().size() << std::endl;
    std::cout << model_d.getKeypoints().getColor().size() << std::endl;

    auto matcher = descry::Matcher<cv::Mat>{};
    matcher.configure(cfg[descry::config::matcher::NODE_NAME]);
    matcher.setModel(views_description);

    auto matches = matcher.match(scene_d);
    std::cout << matches.front()->size() << std::endl;

    cv::Mat homography;
    auto cv_matches = filter(scene_d, model_d, matches.front(), homography, cfg["alignment"]["ransac-inlier-threshold"].as<double>());
    cv::Mat res;
    drawMatches(image.getColorMat(), scene_d.getKeypoints().getColor(), view.image.getColorMat(), model_d.getKeypoints().getColor(), cv_matches, res);
    //drawCorners(get_corners(view), homography, res);
    cv::namedWindow( "matches", cv::WINDOW_AUTOSIZE );
    cv::imshow( "matches" , res );
    cv::waitKey();


    model.getViews().erase(model.getViews().begin(), model.getViews().begin() + 5);
    model.getViews().erase(model.getViews().begin() + 1, model.getViews().end());


    pcl::console::setVerbosityLevel(pcl::console::L_DEBUG);
    descry::Clusterizer clst;
    clst.configure(cfg[descry::config::clusters::NODE_NAME]);

    auto key_frames = std::vector<descry::KeyFrameHandle>{};
    for (auto& descr : views_description)
        key_frames.emplace_back(descr.getKeyFrame());

    clst.setModel(model, key_frames);
    auto clusters = clst.compute(image, scene_d.getKeyFrame(), matches);

    auto& instance_map = test_data.front().second;
    std::cout << "Ground truth" << std::endl;
    std::cout << instance_map[model_name].front() << std::endl;
    std::cout << "Found" << std::endl;
    for (auto idx = 0u; idx < clusters.poses.size(); ++idx) {
        std::cout << clusters.poses[idx] << std::endl;
        view_projection(image, model, clusters.poses[idx]);

    }
}

int main(int argc, char * argv[]) {
    // parse config
    descry::Config cfg;
    try {
        if (argc > 1)
            cfg = YAML::LoadFile(argv[1]);
    } catch (...) { }

    if (cfg.IsNull()) {
        std::cerr << "No config provided" << std::endl;
        return EXIT_FAILURE;
    }

    recognize(cfg);
    cv::waitKey(1000);

    return EXIT_SUCCESS;
}

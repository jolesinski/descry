#include <chrono>
#include <iostream>

#include <descry/recognizer.h>
#include <descry/willow.h>
#include <descry/test/config.h>
#include <descry/test/data.h>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

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

template <typename ScenePoint, typename ModelPoint = ScenePoint>
void view_projection(const typename pcl::PointCloud<ModelPoint>::ConstPtr& scene,
                     const typename pcl::PointCloud<ModelPoint>::ConstPtr& model,
                     const descry::Pose& pose) {
    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
    viewer.setBackgroundColor (0, 0, 0);
    auto transformed = descry::make_cloud<ModelPoint>();
    pcl::transformPointCloud(*model, *transformed, pose);

    viewer.addPointCloud(scene, "scene");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "scene");

    viewer.addPointCloud(transformed, "model");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "model");

    viewer.addCoordinateSystem (.3);
    viewer.initCameraParameters ();

    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }

    viewer.close();
}

void log_pose_metrics(const descry::Pose& pose, const descry::Pose& ground_truth) {
    std::cout << pose << std::endl;

    Eigen::Vector4f test;
    test << 1, 1, 1, 0;
    std::cout << "Rotation metric: " << (pose * test - ground_truth * test).norm() << std::endl;
    test << 0, 0, 0, 1;
    std::cout << "Translation metric: " << (pose * test - ground_truth * test).norm() << std::endl;
    test << 0.1, 0.1, 0.1, 1;
    std::cout << "Combined metric: " << (pose * test - ground_truth * test).norm() << std::endl;
}

void recognize(const descry::Config& cfg) {
    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    auto test_name = cfg[descry::config::SCENE_NODE].as<std::string>();
    auto model_name = cfg[descry::config::MODEL_NODE].as<std::string>();
    auto test_data = willow.loadSingleTest(test_name, 1);

    auto& instance_map = test_data.front().second;
    auto ground_truth = instance_map[model_name].front();

    auto scene = test_data.front().first;
    auto model = willow.loadModel(model_name);

    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    auto recognizer = descry::Recognizer{};
    recognizer.configure(cfg[descry::config::RECOGNIZER_NODE]);

    auto start = std::chrono::steady_clock::now();
    recognizer.train(model);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::steady_clock::now() - start);
    std::cout << "Training took " << duration.count() << "ms" << std::endl;
    start = std::chrono::steady_clock::now();

    auto instances = recognizer.compute(scene);

    duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::steady_clock::now() - start);
    std::cout << "Recognition took " << duration.count() << "ms" << std::endl;

//    auto start = std::chrono::steady_clock::now();

//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
//            (std::chrono::steady_clock::now() - start);
//    std::cout << "Alignment took " << duration.count() << "ms" << std::endl;
//    start = std::chrono::steady_clock::now();

    // Refinement
//    std::cout << "ICP Refinement" << std::endl;

//    duration = std::chrono::duration_cast<std::chrono::milliseconds>
//            (std::chrono::steady_clock::now() - start);
//    std::cout << "Refinement took " << duration.count() << "ms" << std::endl;
//    start = std::chrono::steady_clock::now();

    // Verification
//    std::cout << "\n\nVerification" << std::endl;

//    duration = std::chrono::duration_cast<std::chrono::milliseconds>
//            (std::chrono::steady_clock::now() - start);

//    std::cout << "Verification took " << duration.count() << "ms" << std::endl;

    std::cout << "Ground truth" << std::endl;
    std::cout << ground_truth << std::endl;
    std::cout << "Found: " << instances.poses.size() << std::endl;
    for (const auto& pose : instances.poses) {
        log_pose_metrics(pose, ground_truth);
        if (cfg["metrics"]["visualize"].as<bool>()) {
            view_projection<descry::FullPoint>(scene, model.getFullCloud(), pose);
        }
    }
}

int main(int argc, char * argv[]) {
    // parse config
    descry::Config cfg;
    try {
        if (argc > 1)
            cfg = YAML::LoadFile(argv[1]);
    } catch (YAML::ParserException& error) {
        std::cerr << "Parser error: " << error.what() << std::endl;
    } catch (YAML::BadFile& error) {
        std::cerr << "Bad config file: " << error.what() << std::endl;
    }

    if (cfg.IsNull()) {
        std::cerr << "Invalid config" << std::endl;
        return EXIT_FAILURE;
    }

    recognize(cfg);

    return EXIT_SUCCESS;
}

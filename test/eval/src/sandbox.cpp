#include <iostream>

#include <descry/descriptors.h>
#include <descry/normals.h>
#include <descry/willow.h>
#include <descry/test/config.h>
#include <descry/test/data.h>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

void view_keys(const descry::Image& image, std::string window_name) {
    //copy to opencv mat
    auto frame = image.getColorMat();
    auto keys = image.getKeypoints().getColor();

    std::cout << "Detected " << keys.size() << std::endl;

    cv::drawKeypoints(frame, keys, frame, cv::Scalar(0,0,128), cv::DrawMatchesFlags::DEFAULT);

    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    cv::imshow( window_name, frame );
}

void view_keys(const descry::Model& model, std::string window_name) {
    //auto keys = compute_harris_3d(image);
    //auto keys = compute_iss_3d(image);

    //copy to opencv mat
    //for (const auto& view : model.getViews()) {
    {
        const auto& view = model.getViews().at(5);
        auto frame = view.image.getColorMat();
        auto keys = view.image.getKeypoints().getColor();

        std::cout << "Detected " << keys.size() << std::endl;
        view_keys(view.image, "model");
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

void drawCorners(const std::vector<cv::Point2f>& obj_corners, cv::Mat H, cv::InputOutputArray result) {
    std::vector<cv::Point2f> scene_corners;
    perspectiveTransform( obj_corners, scene_corners, H);

    cv::line( result, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4 );
    cv::line( result, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
    cv::line( result, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
    cv::line( result, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );
}

std::vector<cv::DMatch> filter(descry::ColorDescription& scene_d, descry::ColorDescription& model_d, std::vector<cv::DMatch>& matches, cv::Mat& H) {
    //const double ransac_thresh = 2.5f; // RANSAC inlier threshold
    cv::Mat inlier_mask;
    std::vector<cv::KeyPoint> inliers1, inliers2;
    std::vector<cv::DMatch> inlier_matches;

    if (matches.size() < 4) {
        std::cout << "not enough matches" << std::endl;
        return {};
    }

    std::vector<cv::Point2f> scene;
    std::vector<cv::Point2f> obj;

    for( int i = 0; i < matches.size(); i++ )
    {
        scene.push_back( scene_d.keypoints[ matches[i].queryIdx ].pt );
        obj.push_back( model_d.keypoints[ matches[i].trainIdx ].pt );
    }

    H = findHomography(obj, scene, inlier_mask, cv::RANSAC);

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

std::vector<cv::DMatch> match(descry::ColorDescription& scene_d, descry::ColorDescription& model_d, cv::Mat& H) {
    const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector< std::vector<cv::DMatch> > nn_matches;
    matcher.knnMatch(scene_d.descriptors, model_d.descriptors, nn_matches, 2);

    std::vector<cv::DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        cv::DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2)
            good_matches.emplace_back(first);
    }

    return filter(scene_d, model_d, good_matches, H);
}

void recognize(const descry::Config& cfg) {
    auto scene_cfg = cfg["scene"];
    auto model_cfg = cfg["model"];
    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    auto test_name = scene_cfg["name"].as<std::string>();
    auto model_name = model_cfg["name"].as<std::string>();
    auto test_data = willow.loadSingleTest(test_name, 1);
    auto image = descry::Image(test_data.front().first);

    if (cfg["scene"][descry::config::normals::NODE_NAME]) {
        auto nest = descry::NormalEstimation{};
        nest.configure(cfg["scene"][descry::config::normals::NODE_NAME]);
        image.setNormals(nest.compute(image));
    }

    auto describer = descry::Describer<descry::ColorDescription>{};
    describer.configure(cfg["scene"][descry::config::descriptors::NODE_NAME]);
    auto scene_d = describer.compute(image);

//    view_keys(image, "scene");

    // model
    auto model = willow.loadModel(model_name);
    auto& view = model.getViews().at(5);
    //drawViewWithCorners(view);
    describer.configure(cfg["scene"][descry::config::descriptors::NODE_NAME]);
    auto model_d = describer.compute(view.image);
//    view_keys(model, "model");

    cv::Mat homography;
    auto matches = match(scene_d, model_d, homography);
    cv::Mat res;
    drawMatches(image.getColorMat(), scene_d.keypoints, view.image.getColorMat(), model_d.keypoints, matches, res);
    drawCorners(get_corners(view), homography, res);
    cv::namedWindow( "matches", cv::WINDOW_AUTOSIZE );
    cv::imshow( "matches" , res );
    cv::waitKey();
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
    cv::waitKey();

    return EXIT_SUCCESS;
}

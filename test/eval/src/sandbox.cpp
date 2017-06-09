#include <iostream>

#include <descry/recognizer.h>
#include <descry/willow.h>
#include <descry/test/config.h>
#include <descry/test/data.h>

#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <spdlog/fmt/ostr.h>
#include <descry/logger.h>

descry::logger::handle g_log;

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


void log_pose_metrics(const descry::Pose& pose, const descry::Pose& ground_truth) {
    g_log->info("\n{}", pose);

    Eigen::Vector4f test;
    test << 1, 1, 1, 0;
    g_log->info("Rotation metric: {}", (pose * test - ground_truth * test).norm());
    test << 0, 0, 0, 1;
    g_log->info("Translation metric: {}", (pose * test - ground_truth * test).norm());
    test << 0.1, 0.1, 0.1, 1;
    g_log->info("Combined metric: {}", (pose * test - ground_truth * test).norm());
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
    g_log->info("Training took: {}ms", duration.count());
    start = std::chrono::steady_clock::now();

    auto instances = recognizer.compute(scene);

    duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::steady_clock::now() - start);
    g_log->info("Recognition took: {}ms", duration.count());

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

    g_log->info("Ground truth:\n{}", ground_truth);
    g_log->info("Found: {}", instances.poses.size());
    for (const auto& pose : instances.poses)
        log_pose_metrics(pose, ground_truth);
}

int main(int argc, char * argv[]) {
    descry::logger::init();
    g_log = descry::logger::get();
    if (!g_log)
        return EXIT_FAILURE;

    auto cfg = descry::Config{};
    try {
        if (argc > 1)
            cfg = YAML::LoadFile(argv[1]);
    } catch (YAML::ParserException& error) {
        g_log->error("Parser error: {}", error.what());
    } catch (YAML::BadFile& error) {
        g_log->error("Bad config file: {}", error.what());
    }

    if (cfg.IsNull()) {
        g_log->error("Invalid config");
        return EXIT_FAILURE;
    }

    recognize(cfg);

    return EXIT_SUCCESS;
}

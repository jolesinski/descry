#include <benchmark/benchmark.h>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/keypoints.h>
#include <descry/willow.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

static void BM_Keypoints_Time(benchmark::State& state, descry::Config&& cfg) {
    //TODO: load single model, and single test scene with ground truth
    // calculate model views and scene keypoints, extract gt keys from scene keys
    // calculate how many gt keys has a model match within given precision distance
    // Write class WillowTestSet that inherits from WillowDatabase, filter gt points based on pose and full 3d model

    // load model and prepare
    auto model_cfg = descry::Config{};
    model_cfg[descry::config::keypoints::NODE_NAME] = cfg;

    auto prep = descry::Preprocess{};
    prep.configure(model_cfg);

    auto willow = descry::WillowDatabase(descry::test::loadDBConfig()["models"]);
    auto model = willow.loadModel("object_10");
    model.prepare(prep);

    std::cout << "Model prepared" << std::endl;

    auto image = descry::Image(descry::test::loadSceneCloud());
    auto kdet = descry::ShapeKeypointDetector{};

    if(!kdet.configure(cfg))
        state.SkipWithError("Configuration failed");

    double gt_keys = 1;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(kdet.compute(image));
        gt_keys += 1;
    }

    state.counters["repeatability_rate"] = gt_keys;
}
//BENCHMARK_CAPTURE(BM_Keypoints_Time, uniform, descry::test::keypoints::loadConfigUniform())->Unit(benchmark::kMillisecond);
//BENCHMARK_CAPTURE(BM_Keypoints, iss, descry::test::keypoints::loadConfigISS())->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(BM_Keypoints_Time, iss_cupcl, descry::test::keypoints::loadConfigCupcl())->Unit(benchmark::kMillisecond);

cv::Mat loadSceneFrame() {
    auto image = descry::Image(descry::test::loadSceneCloud());
    const auto& full = image.getFullCloud().host();

    //copy to opencv mat
    typedef cv::Point3_<uint8_t> Pixel;
    cv::Mat frame = cv::Mat::zeros(full->width, full->height, CV_8UC3);
    frame.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
        const auto& point = full->at(full->width - position[0] - 1, position[1]);
        pixel.x = point.b;
        pixel.y = point.g;
        pixel.z = point.r;
    });

    return frame;
}

static void BM_OpenCV_ORB_detectAndCompute(benchmark::State& state) {
    auto frame = loadSceneFrame();
    std::vector<cv::KeyPoint> kp;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->setMaxFeatures(1000);
    cv::Mat desc;
    while (state.KeepRunning())
        orb->detectAndCompute(frame, cv::noArray(), kp, desc);
}
BENCHMARK(BM_OpenCV_ORB_detectAndCompute);

static void BM_OpenCV_ORB_detect_then_compute(benchmark::State& state) {
    auto frame = loadSceneFrame();
    std::vector<cv::KeyPoint> kp;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->setMaxFeatures(1000);
    cv::Mat desc;
    while (state.KeepRunning()) {
        orb->detect(frame, kp);
        orb->compute(frame, kp, desc);
    }
}
BENCHMARK(BM_OpenCV_ORB_detect_then_compute);
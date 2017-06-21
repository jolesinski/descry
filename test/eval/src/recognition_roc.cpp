#include <descry/common.h>
#include <descry/latency.h>
#include <descry/recognizer.h>
#include <descry/test/config.h>
#include <descry/willow.h>

#include <pcl/console/print.h>

class RecognitionROC {
    RecognitionROC() : log_(descry::logger::get()) {}
private:
    descry::logger::handle log_;
};

void log_pose_metrics(const descry::AlignedVector<descry::Pose>& found_poses,
                      const descry::AlignedVector<descry::Pose>& gt_poses) {

    std::vector<std::pair<int,double>> found_gts(gt_poses.size(), {-1,std::numeric_limits<double>::max()});

    for (auto found_idx = 0u; found_idx < found_poses.size(); ++found_idx) {
        auto min_distance = std::numeric_limits<double>::max();
        auto matched_gt_idx = -1;
        for (auto gt_idx = 0u; gt_idx < gt_poses.size(); ++gt_idx) {
            Eigen::Vector4f test;
            test << 1, 1, 1, 0;
            auto rotation_error = (found_poses[found_idx]*test - gt_poses[gt_idx]*test).norm();
            test << 0, 0, 0, 1;
            auto translation_error = (found_poses[found_idx]*test - gt_poses[gt_idx]*test).norm();
            descry::logger::get()->info("Accuracy: rotation: {} translation: {}", rotation_error, translation_error);

            if (min_distance > translation_error) {
                min_distance = translation_error;
                matched_gt_idx = gt_idx;
            }
        }

        if (matched_gt_idx != -1 && found_gts[matched_gt_idx].second > min_distance) {
            found_gts[matched_gt_idx].first = found_idx;
            found_gts[matched_gt_idx].second = min_distance;
        }
    }

    for (auto tp : found_gts) {
        if (tp.first != -1 && tp.second < 0.02)
            descry::logger::get()->info("TRUE POSITIVE {} {}", tp.first, tp.second);
        else if (tp.first == -1)
            descry::logger::get()->info("FALSE NEGATIVE");
        else
            descry::logger::get()->info("FALSE POSITIVE");
    }
    if (found_poses.empty() && gt_poses.empty())
        descry::logger::get()->info("TRUE NEGATIVE");
}

void recognize(const descry::Config& cfg) {
    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    auto test_name = cfg[descry::config::SCENE_NODE].as<std::string>();
    auto model_name = cfg[descry::config::MODEL_NODE].as<std::string>();
    auto latency = descry::measure_latency("Loading test set");
    auto test_data = willow.loadSingleTest(test_name);
    latency.restart("Loading model");
    auto model = willow.loadModel(model_name);
    latency.finish();


    auto recognizer = descry::Recognizer{};
    recognizer.configure(cfg[descry::config::RECOGNIZER_NODE]);

    latency.start("Training");
    recognizer.train(model);
    latency.finish();

    for (const auto& annotated_scene : test_data) {
        const auto& scene = annotated_scene.first;
        const auto& annotations = annotated_scene.second;

        latency.start("Recognition");
        auto instances = recognizer.compute(scene);
        latency.finish();

        log_pose_metrics(instances.poses, annotations.at(model_name));
    }
}

int main(int argc, char * argv[]) {
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
    descry::logger::init();
    auto g_log = descry::logger::get();
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

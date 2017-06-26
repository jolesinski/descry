#include <descry/common.h>
#include <descry/latency.h>
#include <descry/recognizer.h>
#include <descry/test/config.h>
#include <descry/willow.h>

#include <pcl/console/print.h>

class RecognitionROC {
public:
    struct Stats {
        unsigned int true_positive_ = 0;
        unsigned int true_negative_ = 0;
        unsigned int false_positive_ = 0;
        unsigned int false_negative_ = 0;

        Stats operator+(Stats b) const {
            b.true_positive_ += true_positive_;
            b.true_negative_ += true_negative_;
            b.false_positive_ += false_positive_;
            b.false_negative_ += false_negative_;
            return b;
        }

        double getTruePositiveRate() const {
            if ((true_positive_ + false_negative_) == 0)
                return -1;
            return true_positive_/static_cast<double>(true_positive_ + false_negative_);
        }

        double getFalsePositiveRate() const {
            if ((false_positive_ + true_negative_) == 0)
                return -1;
            return false_positive_/static_cast<double>(false_positive_ + true_negative_);
        }

        double getPositivePredictiveValue() const {
            if ((true_positive_ + false_positive_) == 0)
                return -1;
            return true_positive_/static_cast<double>(true_positive_ + false_positive_);
        }

        double getAccuracy() const {
            auto all_samples = (true_positive_ + false_positive_ + false_negative_ + true_negative_);
            if (all_samples == 0)
                return -1;
            return (true_positive_ + true_negative_)/static_cast<double>(all_samples);
        }

        double getPrecision() const { return getPositivePredictiveValue(); }
        double getRecall() const { return getTruePositiveRate(); }
    };

    RecognitionROC() : log_(descry::logger::get()),
                       willow_db_(descry::test::loadDBConfig()),
                       willow_ts_(descry::test::loadDBConfig()) {}

    static float rotationError(const descry::Pose& found, const descry::Pose& gt) {
        Eigen::Vector4f test;
        test << 1, 1, 1, 0;
        return (found*test - gt*test).norm();
    }

    static float translationError(const descry::Pose& found, const descry::Pose& gt) {
        Eigen::Vector4f test;
        test << 0, 0, 0, 1;
        return (found*test - gt*test).norm();
    }

    std::vector<std::pair<int,double>> matchPoses(const descry::AlignedVector<descry::Pose>& found_poses,
                                                  const descry::AlignedVector<descry::Pose>& gt_poses) {
        std::vector<std::pair<int,double>> found_gts(gt_poses.size(), {-1,std::numeric_limits<double>::max()});

        for (auto found_idx = 0u; found_idx < found_poses.size(); ++found_idx) {
            auto closest_gt = std::min_element(std::begin(gt_poses), std::end(gt_poses),
                                               [found{found_poses[found_idx]}](const auto& gt1, const auto& gt2)
                                               { return translationError(found, gt1) < translationError(found, gt2); });

            if (closest_gt != std::end(gt_poses)) {
                auto& found_gt = found_gts[closest_gt - std::begin(gt_poses)];
                auto rotation = rotationError(found_poses[found_idx], *closest_gt);
                auto translation = translationError(found_poses[found_idx], *closest_gt);

                log_->info("Accuracy: rotation: {} translation: {}", rotation, translation);

                if (found_gt.second > translation) {
                    found_gt.first = found_idx;
                    found_gt.second = translation;
                }
            }
        }

        return found_gts;
    };

    void collectStats(const std::string& model_name,
                      const descry::AlignedVector<descry::Pose>& found_poses,
                      const descry::AlignedVector<descry::Pose>& gt_poses) {
        if (found_poses.empty() && gt_poses.empty()) {
            model_stats_[model_name].true_negative_++;
            return;
        } else if (gt_poses.empty()) {
            model_stats_[model_name].false_positive_ += found_poses.size();
            return;
        }

        auto found_gts = matchPoses(found_poses, gt_poses);

        for (auto tp : found_gts) {
            if (tp.first != -1 && tp.second < max_translation_error_)
                model_stats_[model_name].true_positive_++;
            else if (tp.first == -1)
                model_stats_[model_name].false_negative_++;
            else
                model_stats_[model_name].false_positive_++;
        }
    }

    static void logStats(const Stats& stats) {
        descry::logger::get()->info("TP {} FN {} FP {} TN {}",
                                    stats.true_positive_, stats.false_negative_,
                                    stats.false_positive_, stats.true_negative_);
        descry::logger::get()->info("TPR (recall) {} FPR {} PPV (precision) {} ACC {}",
                                    stats.getRecall(), stats.getFalsePositiveRate(),
                                    stats.getPrecision(), stats.getAccuracy());
    }

    std::vector<std::string> getModelNames(const descry::Config& cfg) {
        auto models_cfg = cfg["models"];
        if (!models_cfg)
            return willow_db_.getModelNames();

        if (models_cfg.IsSequence()) {
            return models_cfg.as<std::vector<std::string>>();
        } else
            DESCRY_THROW(descry::InvalidConfigException, "models node is not a sequence")
    }

    std::vector<std::string> getTestNames(const descry::Config& cfg) {
        auto tests_cfg = cfg["tests"];
        if (!tests_cfg)
            return willow_ts_.getTestNames();

        if (tests_cfg.IsSequence()) {
            return tests_cfg.as<std::vector<std::string>>();
        } else
            DESCRY_THROW(descry::InvalidConfigException, "tests node is not a sequence")
    }

    void evaluate(const descry::Config& cfg) {
        recognizer_.configure(cfg[descry::config::RECOGNIZER_NODE]);

        auto test_names = getTestNames(cfg);
        auto model_names = getModelNames(cfg);

        auto progress_count = 0u;
        auto progress_total = model_names.size() * test_names.size();

        for (const auto& model_name : model_names) {
            log_->info("Processing model {}", model_name);
            auto latency = descry::measure_latency("Loading model");
            auto model = willow_db_.loadModel(model_name);
            latency.start("Training");
            recognizer_.train(model);
            latency.finish();

            for (const auto& test_name : test_names) {
                log_->info("Processing test {}", test_name);
                latency.start("Loading test scenes");
                auto test_data = willow_ts_.loadSingleTest(test_name);
                latency.finish();
                for (const auto& annotated_scene : test_data) {
                    const auto& scene = annotated_scene.first;
                    const auto& annotations = annotated_scene.second;

                    latency.start("Recognition");
                    auto instances = recognizer_.compute(scene);
                    latency.finish();

                    if (annotations.count(model_name))
                        collectStats(model_name, instances.poses, annotations.at(model_name));
                    else
                        collectStats(model_name, instances.poses, {});
                }
                log_->info("Progress {}%", (++progress_count*100)/progress_total);
            }
            logStats(model_stats_[model_name]);
        }
        logStats(std::accumulate(std::begin(model_stats_), std::end(model_stats_), Stats{},
                                 [](const auto& a, const auto& b){ return a + b.second; }));
    }

private:
    descry::logger::handle log_;
    descry::Recognizer recognizer_;
    descry::WillowDatabase willow_db_;
    descry::WillowTestSet willow_ts_;

    double max_translation_error_ = 0.02;

    std::unordered_map<std::string, Stats> model_stats_;
};

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


    auto roc = RecognitionROC{};
    roc.evaluate(cfg);

    return EXIT_SUCCESS;
}

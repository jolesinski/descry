#include <descry/matcher.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/features2d.hpp>

namespace descry {

namespace {
template<class Descriptor>
std::unique_ptr<Matcher<Descriptor>> makeStrategy(const Config& config);
}

template<class D>
bool Matcher<D>::configure(const Config& config) {
    strategy_ = makeStrategy<D>(config);

    return (strategy_ != nullptr);
}

template<class D>
void Matcher<D>::train(const std::vector<Description<D>>& model) {
    if (!strategy_) DESCRY_THROW(NotConfiguredException, "Matcher not configured");
    return strategy_->train(model);
}

template<class D>
void Matcher<D>::train(std::vector<Description<D>>&& model) {
    if (!strategy_) DESCRY_THROW(NotConfiguredException, "Matcher not configured");
    return strategy_->train(std::move(model));
}

template<class D>
std::vector<pcl::CorrespondencesPtr> Matcher<D>::match(const Description<D>& scene) {
    if (!strategy_) DESCRY_THROW(NotConfiguredException, "Matcher not configured");
    return strategy_->match(scene);
}

namespace {

template<class D>
bool descriptorFinite(const D& descr) {
    return std::isfinite(descr.histogram[0]);
}

template<>
bool descriptorFinite(const pcl::SHOT352& descr) {
    return std::isfinite(descr.descriptor[0]);
}

template<class Descriptor>
class KDTreeFlannMatching : public Matcher<Descriptor> {
public:
    using Tree = pcl::KdTreeFLANN<Descriptor>;

    KDTreeFlannMatching(const Config& config) {
        max_distance = config[config::matcher::MAX_DISTANCE].as<float>();

        if (config[config::matcher::MAX_NEIGHS])
            max_neighs = config[config::matcher::MAX_NEIGHS].as <unsigned int>();

        if (config[config::matcher::USE_LOWE]) {
            use_lowe = config[config::matcher::USE_LOWE].as<bool>();
            max_neighs = 2;
        }

        if (config[config::matcher::LOWE_RATIO])
            lowe_ratio = config[config::matcher::LOWE_RATIO].as<double>();
    }

    ~KDTreeFlannMatching() override {};

    void train(const std::vector<Description<Descriptor>>& model) override {
        trees.resize(model.size());
        for (auto idx = 0u; idx < model.size(); ++idx) {
            trees[idx].setInputCloud(model[idx].getFeatures().host());
        }
    }

    void train(std::vector<Description<Descriptor>>&& model) override {
        trees.resize(model.size());
        for (auto idx = 0u; idx < model.size(); ++idx)
            trees[idx].setInputCloud(model[idx].getFeatures().host());
    }

    std::vector<pcl::CorrespondencesPtr> match(const Description<Descriptor>& scene) override {
        std::vector<pcl::CorrespondencesPtr> view_corrs;
        for (const auto& tree : trees)
            view_corrs.emplace_back(match_view(scene.getFeatures(), tree));

        return view_corrs;
    }

    pcl::CorrespondencesPtr match_view(const DescriptorContainer<Descriptor>& scene, const Tree& tree) {
        pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

        // For each scene keypoint descriptor, find nearest neighbor into the model
        // keypoints descriptor cloud and add it to the correspondences vector.
        std::vector<int> neigh_indices(max_neighs);
        std::vector<float> neigh_sqr_dists(max_neighs);

        for (size_t i = 0; i < scene.host()->size(); ++i) {
            if (!descriptorFinite<Descriptor>(scene.host()->at(i))) //skipping NaNs
                continue;

            int found_neighs = tree.nearestKSearch(scene.host()->at(i), 2, neigh_indices, neigh_sqr_dists);

            if (use_lowe && found_neighs == 2
                && neigh_sqr_dists[0] < max_distance
                && neigh_sqr_dists[0] < lowe_ratio * lowe_ratio * neigh_sqr_dists[1]) {
                    model_scene_corrs->emplace_back(neigh_indices[0], static_cast<int>(i), neigh_sqr_dists[0]);
            } else {
                for (int idx = 0; idx < found_neighs; ++idx)
                    if (neigh_sqr_dists[idx] < max_distance)
                        model_scene_corrs->emplace_back(neigh_indices[idx], static_cast<int>(i), neigh_sqr_dists[idx]);
            }
        }

        return model_scene_corrs;
    }

    unsigned int getMaxNeighs() const {
        return max_neighs;
    }

    double getMaxDistance() const {
        return max_distance;
    }

private:
    std::vector<Tree> trees;
    unsigned int max_neighs = 1;
    double max_distance = 0.0;
    bool use_lowe = false;
    double lowe_ratio = 0.8;
};


template<>
class KDTreeFlannMatching<cv::Mat> : public Matcher<cv::Mat> {
public:
    KDTreeFlannMatching(const Config& config) {
        if (config[config::matcher::MAX_DISTANCE])
            max_distance = config[config::matcher::MAX_DISTANCE].as<float>();

        if (config[config::matcher::MAX_NEIGHS])
            max_neighs = config[config::matcher::MAX_NEIGHS].as<unsigned int>();

        if (config[config::matcher::USE_LOWE])
            use_lowe = config[config::matcher::USE_LOWE].as<bool>();

        if (config[config::matcher::LOWE_RATIO])
            lowe_ratio = config[config::matcher::LOWE_RATIO].as<double>();
    }

    void train(const std::vector<Description<cv::Mat>>& model) override {
        model_.clear();
        for (auto& view : model) {
            auto d = Description<cv::Mat>();
            d.setFeatures(view.getFeatures().clone());
            d.setKeypoints(view.getKeypoints());
            model_.emplace_back(std::move(d));
        }
    }

    void train(std::vector<Description<cv::Mat>>&& model) override {
        model_ = std::move(model);
    }

    std::vector<pcl::CorrespondencesPtr> match(const Description<cv::Mat>& scene) override {
        std::vector<pcl::CorrespondencesPtr> view_corrs;
        for (const auto& view_descr : model_)
            view_corrs.emplace_back(match_view(scene, view_descr));

        return view_corrs;
    }

    pcl::CorrespondencesPtr match_view(const Description<cv::Mat>& scene, const Description<cv::Mat>& model) {
        pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

        cv::FlannBasedMatcher matcher;
        std::vector<std::vector<cv::DMatch> > nn_matches;

        if (use_lowe) {
            matcher.knnMatch(scene.getFeatures(), model.getFeatures(), nn_matches, 2);

            for (size_t i = 0; i < nn_matches.size(); i++) {
                cv::DMatch first = nn_matches[i][0];
                float dist1 = nn_matches[i][0].distance;
                float dist2 = nn_matches[i][1].distance;

                if (dist1 < lowe_ratio * dist2)
                    model_scene_corrs->emplace_back(first.trainIdx, first.queryIdx, first.distance);
            }
        } else {
            matcher.knnMatch(scene.getFeatures(), model.getFeatures(), nn_matches, max_neighs);

            for (size_t i = 0; i < nn_matches.size(); ++i) {
                for (unsigned int nn = 0; nn < max_neighs; ++nn) {
                    cv::DMatch match = nn_matches[i][nn];
                    if (match.distance < max_distance)
                        model_scene_corrs->emplace_back(match.trainIdx, match.queryIdx, match.distance);
                }
            }
        }

        return model_scene_corrs;
    }

    ~KDTreeFlannMatching() override = default;
private:
    std::vector<Description<cv::Mat>> model_;
    bool use_lowe = false;
    double lowe_ratio = 0.8;
    double max_distance = 0.0;
    unsigned int max_neighs = 1;
};

class BruteForceMatching : public Matcher<cv::Mat> {
public:
    BruteForceMatching(const Config& config) {
        auto norm_type_str = config[config::matcher::NORM_TYPE].as<std::string>();
        if (norm_type_str == config::matcher::NORM_HAMMING)
            norm_type = cv::NORM_HAMMING;
        else if (norm_type_str == config::matcher::NORM_L2)
            norm_type = cv::NORM_L2;
        else DESCRY_THROW(InvalidConfigException, "Unsupported norm type");

        if (config[config::matcher::MAX_DISTANCE])
            max_distance = config[config::matcher::MAX_DISTANCE].as<float>();

        if (config[config::matcher::MAX_NEIGHS])
            max_neighs = config[config::matcher::MAX_NEIGHS].as<unsigned int>();

        if (config[config::matcher::USE_LOWE])
            use_lowe = config[config::matcher::USE_LOWE].as<bool>();

        if (config[config::matcher::LOWE_RATIO])
            lowe_ratio = config[config::matcher::LOWE_RATIO].as<double>();
    }

    void train(const std::vector<Description<cv::Mat>>& model) override {
        model_.clear();
        for (auto& view : model) {
            auto d = Description<cv::Mat>();
            d.setFeatures(view.getFeatures().clone());
            d.setKeypoints(view.getKeypoints());
            model_.emplace_back(std::move(d));
        }
    }

    void train(std::vector<Description<cv::Mat>>&& model) override {
        model_ = std::move(model);
    }

    std::vector<pcl::CorrespondencesPtr> match(const Description<cv::Mat>& scene) override {
        std::vector<pcl::CorrespondencesPtr> view_corrs;
        for (const auto& view_descr : model_)
            view_corrs.emplace_back(match_view(scene, view_descr));

        return view_corrs;
    }

    pcl::CorrespondencesPtr match_view(const Description<cv::Mat>& scene, const Description<cv::Mat>& model) {
        pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

        cv::BFMatcher matcher(norm_type);
        std::vector<std::vector<cv::DMatch> > nn_matches;

        if (use_lowe) {
            matcher.knnMatch(scene.getFeatures(), model.getFeatures(), nn_matches, 2);

            for (size_t i = 0; i < nn_matches.size(); i++) {
                cv::DMatch first = nn_matches[i][0];
                float dist1 = nn_matches[i][0].distance;
                float dist2 = nn_matches[i][1].distance;

                if (dist1 < lowe_ratio * dist2)
                    model_scene_corrs->emplace_back(first.trainIdx, first.queryIdx, first.distance);
            }
        } else {
            matcher.knnMatch(scene.getFeatures(), model.getFeatures(), nn_matches, max_neighs);

            for (size_t i = 0; i < nn_matches.size(); ++i) {
                for (unsigned int nn = 0; nn < max_neighs; ++nn) {
                    cv::DMatch match = nn_matches[i][nn];
                    if (match.distance < max_distance)
                        model_scene_corrs->emplace_back(match.trainIdx, match.queryIdx, match.distance);
                }
            }
        }

        return model_scene_corrs;
    }

    ~BruteForceMatching() override {};
private:
    std::vector<Description<cv::Mat>> model_;
    cv::NormTypes norm_type;
    bool use_lowe = false;
    double lowe_ratio = 0.8;
    double max_distance = 0.0;
    unsigned int max_neighs = 1;
};

template<class Descriptor>
std::unique_ptr<Matcher<Descriptor>> makeStrategy(const Config& config) {
    if (!config.IsMap() || !config["type"])
        return nullptr;

    try {
        auto descr_type = config["type"].as<std::string>();
        if (descr_type == config::matcher::KDTREE_FLANN_TYPE)
            return std::make_unique<KDTreeFlannMatching<Descriptor>>(config);
    } catch (const YAML::RepresentationException& e) {}

    return nullptr;
}

template<>
std::unique_ptr<Matcher<cv::Mat>> makeStrategy<cv::Mat>(const Config& config) {
    if (!config.IsMap() || !config["type"])
        return nullptr;

    try {
        auto descr_type = config["type"].as<std::string>();
        if (descr_type == config::matcher::BRUTE_FORCE_TYPE)
            return std::make_unique<BruteForceMatching>(config);
        else if (descr_type == config::matcher::KDTREE_FLANN_TYPE)
            return std::make_unique<KDTreeFlannMatching<cv::Mat>>(config);
    } catch (const YAML::RepresentationException& e) {}

    return nullptr;
}

}

template
class Matcher<pcl::SHOT352>;

template
class Matcher<pcl::FPFHSignature33>;

template
class Matcher<cv::Mat>;

}
#include <descry/matching.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace descry {

template <class Descriptor>
using MatcherStrategy = typename Matcher<Descriptor>::Strategy;

template <class Descriptor>
std::unique_ptr<MatcherStrategy<Descriptor>> makeStrategy(const Config& config);

template<class D>
bool Matcher<D>::configure(const Config& config) {
    strategy_ = makeStrategy<D>(config);

    return (strategy_ != nullptr);
}

template<class D>
void Matcher<D>::setModel(const std::vector<DualDescriptors>& model) {
    assert(strategy_);
    return strategy_->setModel(model);
}

template<class D>
std::vector<pcl::CorrespondencesPtr> Matcher<D>::match(const DualDescriptors& scene) {
    assert(strategy_);
    return strategy_->match(scene);
}

template <class D>
bool descriptorFinite(const D& descr) {
    return std::isfinite(descr.histogram[0]);
}

template <>
bool descriptorFinite(const pcl::SHOT352& descr) {
    return std::isfinite(descr.descriptor[0]);
}

template <class Descriptor>
class KDTreeFlannMatching : public MatcherStrategy<Descriptor> {
public:
    using DualDescriptors = typename Matcher<Descriptor>::DualDescriptors;
    using Tree = pcl::KdTreeFLANN<Descriptor>;

    KDTreeFlannMatching(const Config& config) {
        max_distance = config[config::matcher::MAX_DISTANCE].as<unsigned int>();

        if (config[config::matcher::MAX_NEIGHS])
            max_neighs = config[config::matcher::MAX_NEIGHS].as<unsigned int>();
    }

    ~KDTreeFlannMatching() override {};

    void setModel(const std::vector<DualDescriptors>& model) override {
        trees.resize(model.size());
        for (auto idx = 0u; idx < model.size(); ++idx)
            trees[idx].setInputCloud(model[idx].host());
    }

    std::vector<pcl::CorrespondencesPtr> match(const DualDescriptors& scene) override {
        std::vector<pcl::CorrespondencesPtr> view_corrs;
        for (const auto& tree : trees)
            view_corrs.emplace_back(match_view(scene, tree));
    }

    pcl::CorrespondencesPtr match_view(const DualDescriptors& scene, const Tree& tree) {
        pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

        // For each scene keypoint descriptor, find nearest neighbor into the model
        // keypoints descriptor cloud and add it to the correspondences vector.
        for (size_t i = 0; i < scene.host()->size (); ++i)
        {
            std::vector<int> neigh_indices(max_neighs);
            std::vector<float> neigh_sqr_dists(max_neighs);
            if (!descriptorFinite<Descriptor>(scene.host()->at(i))) //skipping NaNs
                continue;

            int found_neighs = tree.nearestKSearch(scene.host()->at(i), max_neighs, neigh_indices, neigh_sqr_dists);

            for (int idx = 0; idx < found_neighs; ++idx)
                if(neigh_sqr_dists[idx] < max_distance)
                    model_scene_corrs->emplace_back(neigh_indices[idx], static_cast<int>(i), neigh_sqr_dists[idx]);
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
};

template <class Descriptor>
std::unique_ptr<MatcherStrategy<Descriptor>> makeStrategy(const Config& config) {
    if(!config["matcher"]["type"])
        return nullptr;

    auto descr_type = config["matcher"]["type"].as<std::string>();
    if (descr_type == config::matcher::KDTREE_FLANN_TYPE)
        return std::make_unique<KDTreeFlannMatching<Descriptor>>(config);
    else
        return nullptr;
}

template
class Matcher<pcl::SHOT352>;

template
class Matcher<pcl::FPFHSignature33>;

}
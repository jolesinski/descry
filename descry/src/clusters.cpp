#include <descry/clusters.h>

// using precompiled correspondence rejector causes stack smashing in umeyama jacobi svd (?)
#define PCL_NO_PRECOMPILE
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>

namespace {
using CG_Hough = pcl::Hough3DGrouping<descry::ShapePoint, descry::ShapePoint>;
using CG_Geo = pcl::GeometricConsistencyGrouping<descry::ShapePoint, descry::ShapePoint>;
}

using namespace descry::config::clusters;

namespace YAML {
template<>
struct convert<CG_Geo> {
    static bool decode(const Node &node, CG_Geo &rhs) {
        if (!node.IsMap())
            return false;

        // required
        if (!node[GC_SIZE] || !node[GC_THRESH])
            return false;

        rhs.setGCSize(node[GC_SIZE].as<double>());
        rhs.setGCThreshold(node[GC_THRESH].as<int>());
        return true;
    }
};

template<>
struct convert<CG_Hough> {
    static bool decode(const Node &node, CG_Hough &rhs) {
        if (!node.IsMap())
            return false;

        // required
        if (!node[BIN_SIZE] || !node[HOUGH_THRESH])
            return false;

        rhs.setHoughBinSize(node[BIN_SIZE].as<double>());
        rhs.setHoughThreshold(node[HOUGH_THRESH].as<double>());

        // optionals
        {
            auto &elem = node[USE_INTERP];
            if (elem)
                rhs.setUseInterpolation(elem.as<bool>());
        }

        {
            auto &elem = node[USE_DIST_W];
            if (elem)
                rhs.setUseDistanceWeight(elem.as<bool>());
        }

        return true;
    }
};
}

namespace descry {

std::unique_ptr<Clusterizer::Strategy> makeStrategy(const Config& config);

bool Clusterizer::configure(const Config &config) {
    strategy_ = makeStrategy(config);

    return (strategy_ != nullptr);
}

void Clusterizer::train(const Model& model, const std::vector<KeyFrameHandle>& view_keyframes) {
    if (!strategy_)
        DESCRY_THROW(NotConfiguredException, "Clusterer not configured");
    return strategy_->train(model, view_keyframes);
}

Instances Clusterizer::compute(const Image& image, const KeyFrameHandle& keyframe,
                               const std::vector<pcl::CorrespondencesPtr>& corrs) {
    if (!strategy_)
        DESCRY_THROW(NotConfiguredException, "Clusterer not configured");
    return strategy_->compute(image, keyframe, corrs);
}

template <typename CG>
class PCLStrategy : public Clusterizer::Strategy {
public:
    PCLStrategy(const Config& config) : clust_{config.as<CG>()} {}

    ~PCLStrategy() override {};

    void train(const Model& model, const std::vector<KeyFrameHandle>& view_keyframes) override {
        model_ = model.getFullCloud();
        clust_.resize(model.getViews().size(), clust_.front());

        for (auto idx = 0u; idx < clust_.size(); ++idx)
            clust_[idx].setInputCloud(view_keyframes.at(idx).keys->getShape().host());

        setModelRefFrames(view_keyframes);

        for (const auto& view : model.getViews())
            viewpoints_.emplace_back(view.viewpoint);

        train();
    }

    Instances compute(const Image& /*image*/, const KeyFrameHandle& keyframe,
                      const std::vector<pcl::CorrespondencesPtr>& corrs) override {
        assert(corrs.size() == clust_.size());

        auto instances = Instances{};
        instances.cloud = model_;

        auto poses = AlignedVector<Pose>{};
        for (auto idx = 0u; idx < corrs.size(); ++idx) {
            clust_[idx].setSceneCloud(keyframe.keys->getShape().host());
            setSceneRefFrames(keyframe, idx);
            clust_[idx].setModelSceneCorrespondences(corrs[idx]);
            clust_[idx].recognize(poses);

            for (auto pose : poses)
                instances.poses.emplace_back(pose * viewpoints_[idx].inverse());
        }

        return instances;
    }

    void train() {}
    void setModelRefFrames(const std::vector<KeyFrameHandle>& /*model*/) {}
    void setSceneRefFrames(const KeyFrameHandle& /*image*/, unsigned /*model_idx*/) {}

private:
    FullCloud::ConstPtr model_;
    AlignedVector<Pose> viewpoints_;
    std::vector<CG> clust_;
};

template<>
void PCLStrategy<CG_Hough>::train() {
    for (auto& hough : clust_)
        hough.train();
}

template<>
void PCLStrategy<CG_Hough>::setModelRefFrames(const std::vector<KeyFrameHandle>& keyframes) {
    for (auto idx = 0u; idx < clust_.size(); ++idx)
        clust_[idx].setInputRf(keyframes.at(idx).rfs->host());
}

template<>
void PCLStrategy<CG_Hough>::setSceneRefFrames(const KeyFrameHandle& keyframe, unsigned model_idx) {
    clust_.at(model_idx).setSceneRf(keyframe.rfs->host());
}

std::unique_ptr<Clusterizer::Strategy> makeStrategy(const Config& config) {
    if(!config["type"])
        return nullptr;

    auto clust_type = config["type"].as<std::string>();
    if (clust_type == GEO_TYPE) {
        return std::make_unique<PCLStrategy<CG_Geo>>(config);
    } else if (clust_type == HOUGH_TYPE) {
        return std::make_unique<PCLStrategy<CG_Hough>>(config);
    }
    return nullptr;
}

}
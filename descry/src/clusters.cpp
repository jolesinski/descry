#include <descry/clusters.h>
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

void Clusterizer::setModel(const Model& model) {
    assert(strategy_);
    return strategy_->setModel(model);
}

Instances Clusterizer::compute(const Image& image, const std::vector<pcl::CorrespondencesPtr>& corrs) {
    assert(strategy_);
    return strategy_->compute(image, corrs);
}

template <class CG>
class PCLStrategy : public Clusterizer::Strategy {
public:
    PCLStrategy(const Config& config) : clust_{config.as<CG>()} {}

    ~PCLStrategy() override {};

    void setModel(const Model& model) override {
        model_ = model.getFullCloud();
        clust_.resize(model.getViews().size(), clust_.front());

        for (auto idx = 0u; idx < clust_.size(); ++idx)
            clust_[idx].setInputCloud(model.getViews()[idx].image.getShapeCloud().host());

        setModelRefFrames(model);

        for (const auto& view : model.getViews())
            viewpoints_.emplace_back(view.viewpoint);

        train();
    }

    Instances compute(const Image& image, const std::vector<pcl::CorrespondencesPtr>& corrs) override {
        assert(corrs.size() == clust_.size());
        std::vector<pcl::CorrespondencesPtr> view_corrs;

        auto instances = Instances{};
        instances.cloud = model_;

        setSceneRefFrames(image);
        auto poses = AlignedVector<Pose>{};
        for (auto idx = 0u; idx < corrs.size(); ++idx) {
            clust_[idx].setSceneCloud(image.getShapeCloud().host());
            clust_[idx].setModelSceneCorrespondences(corrs[idx]);
            clust_[idx].recognize(poses);
            for (auto pose : poses)
                instances.poses.emplace_back(pose * viewpoints_[idx]);
        }

        return instances;
    }

    void train() {}
    void setModelRefFrames(const Model& /*model*/) {}
    void setSceneRefFrames(const Image& /*image*/) {}

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
void PCLStrategy<CG_Hough>::setModelRefFrames(const Model& model) {
    for (auto idx = 0u; idx < clust_.size(); ++idx)
        clust_[idx].setInputRf(model.getViews()[idx].image.getRefFrames().host());
}

template<>
void PCLStrategy<CG_Hough>::setSceneRefFrames(const Image& image) {
    for (auto& hough : clust_)
        hough.setSceneRf(image.getRefFrames().host());
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
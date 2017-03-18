#include <descry/alignment.h>
#include <descry/clusters.h>
#include <descry/descriptors.h>
#include <descry/matching.h>

namespace descry {

std::unique_ptr<Aligner::AlignmentStrategy> makeSparseStrategy(const Config& config);

bool Aligner::configure(const Config& config) {
    if (!config["type"])
        return false;

    try {
        auto est_type = config["type"].as<std::string>();
        if (est_type == config::aligner::SPARSE_TYPE)
            strategy_ = makeSparseStrategy(config);
    } catch ( const YAML::RepresentationException& e) {
        return false;
    }

    return (strategy_ != nullptr);
}

void Aligner::setModel(const Model& model) {
    if (!strategy_)
        DESCRY_THROW(NotConfiguredException, "Aligner not configured");
    return strategy_->setModel(model);
}

Instances Aligner::compute(const Image& image) {
    if (!strategy_)
        DESCRY_THROW(NotConfiguredException, "Aligner not configured");
    return strategy_->match(image);
}

template <class Descriptor>
class SparseShapeMatching : public Aligner::AlignmentStrategy {
public:
    SparseShapeMatching(const Config& config);
    ~SparseShapeMatching() override {};

    void setModel(const Model& model);
    Instances match(const Image& image) override;
private:
    Describer<Descriptor> scene_describer;
    Describer<Descriptor> model_describer;

    Matcher<Descriptor> matcher;
    std::vector<Model::Description<Descriptor>> model_description;
    Clusterizer clusterizer;
};

template<class Descriptor>
SparseShapeMatching<Descriptor>::SparseShapeMatching(const Config& config) {
    scene_describer.configure(config[config::descriptors::NODE_NAME]);
    model_describer.configure(config[config::descriptors::NODE_NAME]);

    matcher.configure(config[config::matcher::NODE_NAME]);
    clusterizer.configure(config[config::clusters::NODE_NAME]);
}

template<class Descriptor>
void SparseShapeMatching<Descriptor>::setModel(const Model& model) {
    for(const auto& view : model.getViews())
        model_description.emplace_back(model_describer.compute(view.image));
    matcher.setModel(model_description);
    clusterizer.setModel(model);
}

template<class Descriptor>
Instances SparseShapeMatching<Descriptor>::match(const Image& image) {
    auto scene_descriptors = scene_describer.compute(image);
    auto matches_per_view = matcher.match(scene_descriptors);

    return clusterizer.compute(image, matches_per_view);
}

std::unique_ptr<Aligner::AlignmentStrategy> makeSparseStrategy(const Config& config) {
    if(!config[config::descriptors::NODE_NAME]["type"])
        return nullptr;

    auto descr_type = config[config::descriptors::NODE_NAME]["type"].as<std::string>();
    if (descr_type == config::descriptors::SHOT_PCL_TYPE)
        return std::make_unique<SparseShapeMatching<pcl::SHOT352>>(config);
    else if (descr_type == config::descriptors::FPFH_PCL_TYPE)
        return std::make_unique<SparseShapeMatching<pcl::SHOT352>>(config);
    else
        return nullptr;
}

}

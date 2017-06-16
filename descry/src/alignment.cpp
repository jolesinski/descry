#include <descry/alignment.h>
#include <descry/clusters.h>
#include <descry/descriptors.h>
#include <descry/matcher.h>
#include <descry/viewer.h>

namespace descry {

std::unique_ptr<Aligner::AlignmentStrategy> makeSparseStrategy(const Config& config);

void Aligner::configure(const Config& config) {
    if (!config["type"])
        DESCRY_THROW(InvalidConfigException, "missing aligner type");

    try {
        auto est_type = config["type"].as<std::string>();
        if (est_type == config::aligner::SPARSE_TYPE)
            strategy_ = makeSparseStrategy(config);
    } catch ( const YAML::RepresentationException& e) {
        DESCRY_THROW(InvalidConfigException, e.what());
    }
}

void Aligner::train(const Model& model) {
    if (!strategy_)
        DESCRY_THROW(NotConfiguredException, "Aligner not configured");
    return strategy_->train(model);
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

    void train(const Model& model);
    Instances match(const Image& image) override;
private:
    Describer<Descriptor> scene_describer;
    Describer<Descriptor> model_describer;

    Matcher<Descriptor> matcher;
    std::vector<Description<Descriptor>> model_description;
    Clusterizer clusterizer;
    Viewer<Aligner> viewer;
};

template<class Descriptor>
SparseShapeMatching<Descriptor>::SparseShapeMatching(const Config& cfg) {
    viewer.configure(cfg);

    model_describer.configure(cfg[config::aligner::DESCRIPTION_NODE][config::MODEL_NODE]);
    scene_describer.configure(cfg[config::aligner::DESCRIPTION_NODE][config::SCENE_NODE]);

    matcher.configure(cfg[config::matcher::NODE_NAME]);
    clusterizer.configure(cfg[config::clusters::NODE_NAME]);
}

template<class Descriptor>
void SparseShapeMatching<Descriptor>::train(const Model& model) {
    for(const auto& view : model.getViews())
        model_description.emplace_back(model_describer.compute(view.image));
    matcher.train(model_description);

    auto key_frames = std::vector<KeyFrame::Ptr>{};
    for (auto& descr : model_description)
        key_frames.emplace_back(descr.getKeyFrame());
    clusterizer.train(model, key_frames);
}

template<class Descriptor>
Instances SparseShapeMatching<Descriptor>::match(const Image& image) {
    auto scene_description = scene_describer.compute(image);
    auto matches_per_view = matcher.match(scene_description);

    auto instances = clusterizer.compute(image, *scene_description.getKeyFrame(), matches_per_view);
    viewer.show(image.getFullCloud().host(), instances);
    return instances;
}

namespace {

std::string getDescriptorName(const Config& config) {
    if (!config[config::MODEL_NODE] || !config[config::SCENE_NODE])
        DESCRY_THROW(InvalidConfigException, "missing descriptors config");

    auto& model_feats_node = config[config::MODEL_NODE][config::features::NODE_NAME];
    auto& scene_feats_node = config[config::SCENE_NODE][config::features::NODE_NAME];

    if (!model_feats_node || !scene_feats_node)
        DESCRY_THROW(InvalidConfigException, "malformed descriptors config, missing features");

    auto& model_type_node = model_feats_node[config::TYPE_NODE];
    auto& scene_type_node = scene_feats_node[config::TYPE_NODE];

    if (!model_type_node || !scene_type_node)
        DESCRY_THROW(InvalidConfigException, "malformed descriptors config, missing type");

    auto model_type_name = model_type_node.as<std::string>();
    auto scene_type_name = scene_type_node.as<std::string>();

    if (model_type_name != scene_type_name)
        DESCRY_THROW(InvalidConfigException, "model and scene descriptor types differ");

    return model_type_name;
}

}

std::unique_ptr<Aligner::AlignmentStrategy> makeSparseStrategy(const Config& config) {
    if(!config[config::aligner::DESCRIPTION_NODE])
        DESCRY_THROW(InvalidConfigException, "missing descriptor type");

    auto descr_type = getDescriptorName(config[config::aligner::DESCRIPTION_NODE]);
    if (descr_type == config::features::SHOT_PCL_TYPE)
        return std::make_unique<SparseShapeMatching<pcl::SHOT352>>(config);
    else if (descr_type == config::features::FPFH_PCL_TYPE)
        return std::make_unique<SparseShapeMatching<pcl::SHOT352>>(config);
    else if (descr_type == config::features::ORB_TYPE)
        return std::make_unique<SparseShapeMatching<cv::Mat>>(config);
    else
        DESCRY_THROW(InvalidConfigException, "unsupported descriptor type");
}

}

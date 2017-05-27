#include <descry/alignment.h>
#include <descry/clusters.h>
#include <descry/descriptors.h>
#include <descry/matching.h>

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
    std::vector<Description<Descriptor>> model_description;
    Clusterizer clusterizer;
};

template<class Descriptor>
SparseShapeMatching<Descriptor>::SparseShapeMatching(const Config& config) {
    model_describer.configure(config[config::descriptors::NODE_NAME]["model"]);
    scene_describer.configure(config[config::descriptors::NODE_NAME]["scene"]);

    matcher.configure(config[config::matcher::NODE_NAME]);
    clusterizer.configure(config[config::clusters::NODE_NAME]);
}

template<class Descriptor>
void SparseShapeMatching<Descriptor>::setModel(const Model& model) {
    for(const auto& view : model.getViews())
        model_description.emplace_back(model_describer.compute(view.image));
    matcher.setModel(model_description);

    auto key_frames = std::vector<KeyFrameHandle>{};
    for (auto& descr : model_description)
        key_frames.emplace_back(descr.getKeyFrame());
    clusterizer.setModel(model, key_frames);
}

template<class Descriptor>
Instances SparseShapeMatching<Descriptor>::match(const Image& image) {
    auto scene_description = scene_describer.compute(image);
    auto matches_per_view = matcher.match(scene_description);

    return clusterizer.compute(image, scene_description.getKeyFrame(), matches_per_view);
}

namespace {

std::string getDescriptorName(const Config& config) {
    if (!config["model"] || !config["scene"])
        DESCRY_THROW(InvalidConfigException, "missing descriptors config");

    auto& model_type_node = config["model"]["type"];
    auto& scene_type_node = config["scene"]["type"];

    if (!model_type_node || !scene_type_node)
        DESCRY_THROW(InvalidConfigException, "malformed descriptors config, missing type");

    auto model_type_name = config["model"]["type"].as<std::string>();
    auto scene_type_name = config["scene"]["type"].as<std::string>();

    if (model_type_name != scene_type_name)
        DESCRY_THROW(InvalidConfigException, "model and scene descriptor types differ");

    return model_type_name;
}

}

std::unique_ptr<Aligner::AlignmentStrategy> makeSparseStrategy(const Config& config) {
    if(!config[config::descriptors::NODE_NAME])
        DESCRY_THROW(InvalidConfigException, "missing descriptor type");

    auto descr_type = getDescriptorName(config[config::descriptors::NODE_NAME]);
    if (descr_type == config::descriptors::SHOT_PCL_TYPE)
        return std::make_unique<SparseShapeMatching<pcl::SHOT352>>(config);
    else if (descr_type == config::descriptors::FPFH_PCL_TYPE)
        return std::make_unique<SparseShapeMatching<pcl::SHOT352>>(config);
    else if (descr_type == config::descriptors::ORB_TYPE)
        return std::make_unique<SparseShapeMatching<cv::Mat>>(config);
    else
        DESCRY_THROW(InvalidConfigException, "unsupported descriptor type");
}

}

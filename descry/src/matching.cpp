#include <descry/matching.h>
#include <descry/matcher.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/features2d.hpp>

namespace descry {

namespace {
std::unique_ptr<Matching> makeStrategy(const Config& config);
}

void Matching::configure(const Config& config) {
    try {
        strategy_ = makeStrategy(config);
    } catch ( const YAML::RepresentationException& e) {
        DESCRY_THROW(InvalidConfigException, e.what());
    }
}

std::vector<KeyFrame::Ptr> Matching::train(const Model& model) {
    if (!strategy_)
        DESCRY_THROW(NotConfiguredException, "Matching not configured");
    return strategy_->train(model);
}

ModelSceneMatches Matching::match(const Image& image) {
    if (!strategy_)
        DESCRY_THROW(NotConfiguredException, "Matching not configured");
    return strategy_->match(image);
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

template <class Descriptor>
class SparseMatching : public Matching {
public:
    SparseMatching(const Config& config);
    ~SparseMatching() override {};

    std::vector<KeyFrame::Ptr> train(const Model& model) override;
    ModelSceneMatches match(const Image& image) override;
private:
    Describer<Descriptor> scene_describer;
    Describer<Descriptor> model_describer;

    Matcher<Descriptor> matcher;
    std::vector<Description<Descriptor>> model_description;
    Viewer<Aligner> viewer;
};

template<class Descriptor>
SparseMatching<Descriptor>::SparseMatching(const Config& cfg) {
    viewer.configure(cfg);

    model_describer.configure(cfg[config::matcher::DESCRIPTION_NODE][config::MODEL_NODE]);
    scene_describer.configure(cfg[config::matcher::DESCRIPTION_NODE][config::SCENE_NODE]);
    matcher.configure(cfg[config::matcher::MATCHER_NODE]);
}

template<class Descriptor>
std::vector<KeyFrame::Ptr> SparseMatching<Descriptor>::train(const Model& model) {
    model_description.clear();
    for (const auto& view : model.getViews())
        model_description.emplace_back(model_describer.compute(view.image));
    matcher.train(model_description);

    auto key_frames = std::vector<KeyFrame::Ptr>{};
    for (auto& descr : model_description)
        key_frames.emplace_back(descr.getKeyFrame());

    return key_frames;
}


template<class Descriptor>
ModelSceneMatches SparseMatching<Descriptor>::match(const Image& image) {
    auto scene_description = scene_describer.compute(image);
    auto matches_per_view = matcher.match(scene_description);

    auto model_scene_matches = ModelSceneMatches{};
    model_scene_matches.scene = scene_description.getKeyFrame();
    for (auto idx = 0u; idx < model_description.size(); ++idx)
        model_scene_matches.view_corrs.push_back({matches_per_view.at(idx),
                                             model_description.at(idx).getKeyFrame()});

    return model_scene_matches;
}

std::unique_ptr<Matching> makeStrategy(const Config& cfg) {
    if (!cfg.IsMap())
        DESCRY_THROW(InvalidConfigException, "invalid config");

    if (!cfg[config::matcher::DESCRIPTION_NODE])
        DESCRY_THROW(InvalidConfigException, "missing descriptor type");
    auto descr_type = getDescriptorName(cfg[config::matcher::DESCRIPTION_NODE]);
    if (descr_type == config::features::SHOT_PCL_TYPE)
        return std::make_unique<SparseMatching<pcl::SHOT352>>(cfg);
    else if (descr_type == config::features::FPFH_PCL_TYPE)
        return std::make_unique<SparseMatching<pcl::FPFHSignature33>>(cfg);
    else if (descr_type == config::features::ORB_TYPE)
        return std::make_unique<SparseMatching<cv::Mat>>(cfg);
    else if (descr_type == config::features::AKAZE_TYPE)
        return std::make_unique<SparseMatching<cv::Mat>>(cfg);
    else if (descr_type == config::features::SIFT_TYPE)
        return std::make_unique<SparseMatching<cv::Mat>>(cfg);
    else
        DESCRY_THROW(InvalidConfigException, "unsupported descriptor type");
}

}

}
#include <descry/alignment.h>
#include <boost/make_shared.hpp>
#include <descry/latency.h>

namespace descry {

namespace {
std::unique_ptr<Aligner> makeStrategy(const Config& config);
}

void Aligner::configure(const Config& config) {
    try {
        strategy_ = makeStrategy(config);
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
    return strategy_->compute(image);
}

class SparseAligner : public Aligner {
public:
    SparseAligner(const Config& cfg);

    void train(const Model& model) override;
    Instances compute(const Image& image) override;
private:
    std::vector<KeyFrame::Ptr> folded_kfs_;
    std::vector<Matching> matchings_;
    Clusterizer clustering_;
    Viewer<Aligner> viewer_;
    bool log_latency_ = false;
};

SparseAligner::SparseAligner(const Config& cfg) {
    if (cfg[config::LOG_LATENCY])
        log_latency_ = cfg[config::LOG_LATENCY].as<bool>();

    try {
        auto est_type = cfg[config::TYPE_NODE].as<std::string>();
        if (est_type == config::aligner::SPARSE_TYPE) {
            viewer_.configure(cfg);

            auto matching_cfg = cfg[config::matcher::NODE_NAME];
            matchings_.clear();
            if (matching_cfg.IsMap()) {
                matchings_.emplace_back();
                matchings_.back().configure(matching_cfg);
            } else if (matching_cfg.IsSequence()) {
                for (auto it = matching_cfg.begin(); it != matching_cfg.end(); ++it) {
                    matchings_.emplace_back();
                    matchings_.back().configure(*it);
                }
            } else {
                DESCRY_THROW(InvalidConfigException, "missing matching node");
            }
            clustering_.configure(cfg[config::clusters::NODE_NAME]);
        } else
            DESCRY_THROW(InvalidConfigException, "unsupported aligner type");
    } catch ( const YAML::RepresentationException& e) {
        DESCRY_THROW(InvalidConfigException, e.what());
    }
}

namespace {

std::vector<KeyFrame::Ptr> fold_key_frames(const std::vector<std::vector<KeyFrame::Ptr>>& kfs, const Model& model) {
    auto folded = std::vector<KeyFrame::Ptr>{};
    for (auto view_id = 0u; view_id < kfs.front().size(); ++view_id) {
        auto folded_keys = descry::make_cloud<descry::ShapePoint>();
        auto folded_rfs = descry::make_cloud<pcl::ReferenceFrame>();

        for (auto matcher_id = 0u; matcher_id < kfs.size(); ++matcher_id) {
            folded_keys->reserve(folded_keys->size() + kfs[matcher_id][view_id]->keypoints.size());
            auto& matcher_keys = kfs[matcher_id][view_id]->keypoints.getShape().host();
            folded_keys->insert(folded_keys->end(), matcher_keys->begin(), matcher_keys->end());
            if (!kfs[matcher_id][view_id]->ref_frames.empty()) {
                folded_rfs->reserve(folded_rfs->size() + kfs[matcher_id][view_id]->ref_frames.size());
                auto& matcher_rfs = kfs[matcher_id][view_id]->ref_frames.host();
                folded_rfs->insert(folded_rfs->end(), matcher_rfs->begin(), matcher_rfs->end());
            }
        }

        folded.emplace_back(std::make_shared<KeyFrame>());
        folded.back()->ref_frames = DualRefFrames{std::move(folded_rfs)};
        folded.back()->keypoints = Keypoints{DualShapeCloud{std::move(folded_keys)},
                                                model.getViews().at(view_id).image};
    }

    return folded;
}

ModelSceneMatches fold_matches(const std::vector<ModelSceneMatches>& matches,
                               const Image& image,
                               const std::vector<KeyFrame::Ptr>& model_keys) {
    auto folded = ModelSceneMatches{};
    auto scene_keys = descry::make_cloud<descry::ShapePoint>();
    auto scene_rfs = descry::make_cloud<pcl::ReferenceFrame>();

    for (auto matcher_id = 0u; matcher_id < matches.size(); ++matcher_id) {
        scene_keys->reserve(scene_keys->size() + matches[matcher_id].scene->keypoints.size());
        auto& matcher_keys = matches[matcher_id].scene->keypoints.getShape().host();
        scene_keys->insert(scene_keys->end(), matcher_keys->begin(), matcher_keys->end());
        if (!matches[matcher_id].scene->ref_frames.empty()) {
            scene_rfs->reserve(scene_rfs->size() + matches[matcher_id].scene->ref_frames.size());
            auto& matcher_rfs = matches[matcher_id].scene->ref_frames.host();
            scene_rfs->insert(scene_rfs->end(), matcher_rfs->begin(), matcher_rfs->end());
        }
    }

    folded.scene = std::make_shared<KeyFrame>();
    folded.scene->ref_frames = DualRefFrames{std::move(scene_rfs)};
    folded.scene->keypoints = Keypoints{DualShapeCloud{std::move(scene_keys)}, image};

    auto offset_indices = std::vector<std::pair<int,std::vector<int>>>{{0,std::vector<int>(model_keys.size(), 0)}};
    for (auto idx = 1u; idx < matches.size(); ++idx) {
        offset_indices.emplace_back(offset_indices.at(idx - 1).first + matches[idx - 1].scene->keypoints.size(), std::vector<int>());
        auto& view_offsets = offset_indices.at(idx).second;
        for (auto view_id = 0u; view_id < model_keys.size(); ++view_id) {
            view_offsets.emplace_back(offset_indices.at(idx - 1).second.at(view_id) +
                                      matches[idx - 1].view_corrs.at(view_id).view->keypoints.size());
        }
    }

    for (auto view_id = 0u; view_id < model_keys.size(); ++view_id) {
        folded.view_corrs.emplace_back();
        folded.view_corrs.back().corrs = boost::make_shared<pcl::Correspondences>(*matches.front().view_corrs[view_id].corrs);
        folded.view_corrs.back().view = model_keys[view_id];
    }

    for (auto idx = 1u; idx < matches.size(); ++idx) {
        for (auto view_id = 0u; view_id < model_keys.size(); ++view_id) {
            auto corrs = pcl::Correspondences{};
            corrs.reserve(matches[idx].view_corrs.at(view_id).corrs->size());
            for (auto corr : *matches[idx].view_corrs.at(view_id).corrs)
                corrs.emplace_back(corr.index_query + offset_indices[idx].second.at(view_id),
                                   corr.index_match + offset_indices[idx].first, corr.distance);
            folded.view_corrs[view_id].corrs->insert(folded.view_corrs[view_id].corrs->end(), corrs.begin(), corrs.end());
        }
    }

    return folded;
}

}

void SparseAligner::train(const Model& model) {
    auto key_frames = std::vector<std::vector<KeyFrame::Ptr>>{};
    for (auto& matching : matchings_) {
        key_frames.emplace_back(matching.train(model));
    }

    auto latency = measure_latency("Folding keyframes", log_latency_);
    folded_kfs_ = fold_key_frames(key_frames, model);
    latency.finish();
    clustering_.train(model, folded_kfs_);
}

Instances SparseAligner::compute(const Image& image) {
    auto model_scene_matches = std::vector<ModelSceneMatches>{};
    for (auto& matching : matchings_) {
        auto latency = measure_scope_latency("Matching", log_latency_);
        model_scene_matches.emplace_back(matching.match(image));
    }

    auto latency = measure_latency("Folding matches", log_latency_);
    auto folded_matches = fold_matches(model_scene_matches, image, folded_kfs_);
    latency.finish();

    latency.start("Clustering");
    auto instances = clustering_.compute(image, folded_matches);
    latency.finish();
    viewer_.show(image.getFullCloud().host(), instances);

    logger::get()->debug("Aligner found {}", instances.poses.size());

    return instances;
}

namespace {
std::unique_ptr<Aligner> makeStrategy(const Config& cfg) {
    if (!cfg.IsMap()) DESCRY_THROW(InvalidConfigException, "invalid config");

    if (!cfg[config::TYPE_NODE]) DESCRY_THROW(InvalidConfigException, "missing aligner type");

    auto aligner_type = cfg[config::TYPE_NODE].as<std::string>();
    if (aligner_type == config::aligner::SPARSE_TYPE)
        return std::make_unique<SparseAligner>(cfg);
    else DESCRY_THROW(InvalidConfigException, "unsupported aligner type");
}
}

}

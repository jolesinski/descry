#include <descry/alignment.h>
#include <descry/clusters.h>
#include <descry/matching.h>
#include <descry/segmentation.h>

#include <boost/make_shared.hpp>
#include <descry/latency.h>
#include <opencv2/imgproc.hpp>
#include <pcl/common/centroid.h>

#define PCL_NO_PRECOMPILE
#include <pcl/features/our_cvfh.h>
#undef PCL_NO_PRECOMPILE

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
    log_latency_ = cfg[config::LOG_LATENCY].as<bool>(false);
    viewer_.configure(cfg);

    try {
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
        } else DESCRY_THROW(InvalidConfigException, "missing matching node");
        clustering_.configure(cfg[config::clusters::NODE_NAME]);
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
    logger::get()->info("Found matches: {}", model_scene_matches.size());

    latency.start("Clustering");
    auto instances = clustering_.compute(image, folded_matches);
    latency.finish();
    viewer_.show(image.getFullCloud().host(), instances);

    logger::get()->debug("Aligner found {}", instances.poses.size());

    return instances;
}

class SlidingWindow : public Aligner {
public:
    SlidingWindow(const Config& cfg);

    void train(const Model& model) override;
    Instances compute(const Image& image) override;
private:
    Viewer<Aligner> viewer_;
    bool log_latency_ = false;

    struct Template {
        cv::Mat color;
        cv::Mat mask;
        Pose viewpoint;
    };

    AlignedVector<Template> templates_;
    FullCloud::ConstPtr full_model_;
    ShapePoint model_centroid_;
};

SlidingWindow::SlidingWindow(const Config& cfg) {
    log_latency_ = cfg[config::LOG_LATENCY].as<bool>(false);
    viewer_.configure(cfg);
}

void SlidingWindow::train(const Model& model) {
    full_model_ = model.getFullCloud();
    auto centroid = Eigen::Vector4f{};
    pcl::compute3DCentroid(*full_model_, centroid);
    model_centroid_.getVector4fMap() = centroid;
    for (auto& view : model.getViews()) {
        auto mask = cv::Mat(view.image.getColorMat().size(), CV_8UC3, cv::Scalar(0,0,0));
        auto mask_points = std::vector<cv::Point>{};
        mask_points.reserve(view.mask.size());
        for (auto idx : view.mask) {
            mask_points.emplace_back(idx % view.image.getColorMat().cols, idx / view.image.getColorMat().cols);
            mask.at<cv::Vec3b>(idx)[0] = 255;
            mask.at<cv::Vec3b>(idx)[1] = 255;
            mask.at<cv::Vec3b>(idx)[2] = 255;
        }
        auto rect = cv::boundingRect(mask_points);
        auto cropped = view.image.getColorMat()(rect);
        templates_.emplace_back(Template{cropped, mask(rect),
                                         view.viewpoint});
    }
}

Instances SlidingWindow::compute(const Image& image) {
    auto instances = Instances{};
    instances.cloud = full_model_;
    for (auto temp : templates_) {
        auto result = cv::Mat{};
        cv::matchTemplate(image.getColorMat(), temp.color, result, CV_TM_SQDIFF, temp.mask);
//        cv::normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat{} );
        double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc;
        cv::Point matchLoc;
        cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc);

        if (minVal > 600)
            continue;

        matchLoc = minLoc;
        matchLoc.x += temp.color.cols/2;
        matchLoc.y += temp.color.rows/2;
        // will already know model z offset from scale pyramid
        auto view_centroid = ShapePoint{};
        view_centroid.getVector4fMap() = temp.viewpoint.inverse() * model_centroid_.getVector4fMap();
        auto scene_centroid = ShapePoint{};
        scene_centroid.z = view_centroid.z;

        auto matA = Eigen::Matrix3f{};
        matA.col(0) << matchLoc.x, matchLoc.y, 1;
        matA.col(1) = image.getProjection().host()->block(0,0,3,0);
        matA.col(2) = image.getProjection().host()->block(0,1,3,0);
        auto vecB = Eigen::Vector3f{};
        vecB = (view_centroid.z * image.getProjection().host()->block(0,2,3,0))
               + image.getProjection().host()->block(0,3,3,0);
        Eigen::Vector3f solution = matA.colPivHouseholderQr().solve(vecB);
        scene_centroid.x = solution(1);
        scene_centroid.y = solution(2);

        Eigen::Affine3f view_affine = Eigen::Affine3f::Identity();
        view_affine.matrix() = temp.viewpoint.inverse();
        Eigen::Affine3f affine = Eigen::Affine3f::Identity();
//        affine.translate(view_affine.translation());
        Eigen::Vector3f translation;
        translation.coeffRef(0) = -scene_centroid.x  - model_centroid_.x;
        translation.coeffRef(1) = -scene_centroid.y - model_centroid_.y;
        translation.coeffRef(2) = scene_centroid.z + model_centroid_.z;
        affine.translate(translation);
        affine.rotate(view_affine.rotation());
        instances.poses.emplace_back(affine.matrix());
    }

    viewer_.show(image.getFullCloud().host(), instances);
    return instances;
}

class GlobalAligner : public Aligner {
public:
    GlobalAligner(const Config& cfg);

    void train(const Model& model) override;
    Instances compute(const Image& image) override;
private:
    Segmenter segmenter_;
    FullCloud::ConstPtr full_model_;
    Viewer<Aligner> viewer_;
    std::vector<pcl::PointCloud<pcl::VFHSignature308>::Ptr> view_features_;
    bool log_latency_ = false;
};

GlobalAligner::GlobalAligner(const Config& cfg) {
    segmenter_.configure(cfg[config::segments::NODE_NAME]);

    log_latency_ = cfg[config::LOG_LATENCY].as<bool>(false);
    viewer_.configure(cfg);
}

void GlobalAligner::train(const Model& model) {
    full_model_ = model.getFullCloud();
    segmenter_.train(model);

    for (const auto& view : model.getViews()) {
        pcl::OURCVFHEstimation<ShapePoint, pcl::Normal, pcl::VFHSignature308> ourcvfh;
        ourcvfh.setInputCloud(view.image.getShapeCloud().host());
        ourcvfh.setInputNormals(view.image.getNormals().host());
        ourcvfh.setEPSAngleThreshold(static_cast<float>(5.0 / 180.0 * M_PI)); // 5 degrees.
        ourcvfh.setCurvatureThreshold(1.0);
        ourcvfh.setNormalizeBins(false);
        auto indices_ptr = boost::make_shared<std::vector<int>>(view.mask);
        ourcvfh.setIndices(indices_ptr);
        auto descriptors = make_cloud<pcl::VFHSignature308>();
        ourcvfh.compute(*descriptors);
        view_features_.emplace_back(descriptors);
        logger::get()->error("Training OURCVFH got {} descriptors", descriptors->size());
    }
}

Instances GlobalAligner::compute(const Image& image) {
    auto segments = segmenter_.compute(image);
    logger::get()->error("Segmenter got {} segments", segments.size());

    // OUR-CVFH estimation object.
    pcl::OURCVFHEstimation<ShapePoint, pcl::Normal, pcl::VFHSignature308> ourcvfh;
    ourcvfh.setInputCloud(image.getShapeCloud().host());
    ourcvfh.setInputNormals(image.getNormals().host());
    ourcvfh.setEPSAngleThreshold(static_cast<float>(5.0 / 180.0 * M_PI)); // 5 degrees.
    ourcvfh.setCurvatureThreshold(1.0);
    ourcvfh.setNormalizeBins(false);
    // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
    // this will decide if additional Reference Frames need to be created, if ambiguous.
    ourcvfh.setAxisRatio(0.8);
    for (auto& segment : segments) {
        ourcvfh.setIndices(segment);
        auto latency = measure_scope_latency("OURCVFH", true);
        auto descriptors = make_cloud<pcl::VFHSignature308>();
        ourcvfh.compute(*descriptors);
        logger::get()->error("OURCVFH got {} descriptors", descriptors->size());
    }

    auto instances = Instances{};
    instances.cloud = full_model_;
    return instances;
}

class MockAligner : public Aligner {
    void train(const Model&) override {}
    Instances compute(const Image&) override { return Instances{}; }
};

namespace {
std::unique_ptr<Aligner> makeStrategy(const Config& cfg) {
    if (!cfg.IsMap()) DESCRY_THROW(InvalidConfigException, "invalid config");

    if (!cfg[config::TYPE_NODE]) DESCRY_THROW(InvalidConfigException, "missing aligner type");

    auto aligner_type = cfg[config::TYPE_NODE].as<std::string>();
    if (aligner_type == config::aligner::SPARSE_TYPE)
        return std::make_unique<SparseAligner>(cfg);
    else if (aligner_type == config::aligner::GLOBAL_TYPE)
        return std::make_unique<GlobalAligner>(cfg);
    else if (aligner_type == config::aligner::SLIDING_TYPE)
        return std::make_unique<SlidingWindow>(cfg);
    else if (aligner_type == config::aligner::MOCK_TYPE)
        return std::make_unique<MockAligner>();
    else DESCRY_THROW(InvalidConfigException, "unsupported aligner type");
}
}

}

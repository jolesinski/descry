#include <descry/descriptors.h>

#include <opencv2/features2d.hpp>
#include <descry/latency.h>

#include <opencv2/xfeatures2d/nonfree.hpp>

#define PCL_NO_PRECOMPILE
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>

using SHOT_PCL = pcl::SHOTEstimationOMP<descry::ShapePoint, pcl::Normal, pcl::SHOT352>;
using FPFH_PCL = pcl::FPFHEstimationOMP<descry::ShapePoint, pcl::Normal, pcl::FPFHSignature33>;

namespace YAML {

template <class DescriberPCL>
bool decode_pcl_omp_describer(const YAML::Node& node, DescriberPCL& rhs)
{
    namespace cfg = descry::config::features;
    if (!node.IsMap())
        return false;

    if (node[cfg::SUPPORT_RAD])
        rhs.setRadiusSearch(node[cfg::SUPPORT_RAD].as<double>());
    else
        return false;

    if (node[cfg::THREADS])
        rhs.setNumberOfThreads(node[cfg::THREADS].as<unsigned>());

    return true;
}

template<>
struct convert<SHOT_PCL> {
    static bool decode(const Node &node, SHOT_PCL &rhs) {
        return decode_pcl_omp_describer<SHOT_PCL>(node, rhs);
    }
};

template<>
struct convert<FPFH_PCL> {
    static bool decode(const Node &node, FPFH_PCL &rhs) {
        return decode_pcl_omp_describer<FPFH_PCL>(node, rhs);
    }
};

template<>
struct convert<cv::Ptr<cv::ORB>> {
    static bool decode(const Node &node, cv::Ptr<cv::ORB> &rhs) {
        namespace cfg = descry::config::features;

        if (!node.IsMap())
            return false;

        rhs = cv::ORB::create();

        // optionals
        {
            auto &elem = node[cfg::MAX_FEATURES];
            if (elem)
                rhs->setMaxFeatures(elem.as<int>());
        }

        {
            auto &elem = node[cfg::EDGE_THRESH];
            if (elem)
                rhs->setEdgeThreshold(elem.as<int>());
        }

        {
            auto &elem = node[cfg::FAST_THRESH];
            if (elem)
                rhs->setFastThreshold(elem.as<int>());
        }

        {
            auto &elem = node[cfg::FIRST_LEVEL];
            if (elem)
                rhs->setFirstLevel(elem.as<int>());
        }

        {
            auto &elem = node[cfg::NUM_LEVELS];
            if (elem)
                rhs->setNLevels(elem.as<int>());
        }

        {
            auto &elem = node[cfg::PATCH_SIZE];
            if (elem)
                rhs->setPatchSize(elem.as<int>());
        }

        {
            auto &elem = node[cfg::SCALE_FACTOR];
            if (elem)
                rhs->setScaleFactor(elem.as<int>());
        }

        {
            auto &elem = node[cfg::SCORE_TYPE];
            if (elem)
                rhs->setScoreType(elem.as<int>());
        }

        {
            auto &elem = node[cfg::WTA_K];
            if (elem)
                rhs->setWTA_K(elem.as<int>());
        }

        return true;
    }
};

template<>
struct convert<cv::Ptr<cv::AKAZE>> {
    static bool decode(const Node &node, cv::Ptr<cv::AKAZE> &rhs) {
        namespace cfg = descry::config::features;

        if (!node.IsMap())
            return false;

        rhs = cv::AKAZE::create();

        auto descr_size = node[cfg::AKAZE_SIZE].as<int>(0);
        auto descr_channels = node[cfg::AKAZE_CHANNELS].as<int>(3);
        auto nOctaves = node[cfg::NUM_OCTAVES].as<int>(4);
        auto nOctaveLayers = node[cfg::OCTAVE_LAYERS].as<int>(4);
        auto thresh = node[cfg::AKAZE_THRESH].as<float>(0.001);

        rhs = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, descr_size, descr_channels, thresh, nOctaves, nOctaveLayers);

        return true;
    }
};

template<>
struct convert<cv::Ptr<cv::xfeatures2d::SIFT>> {
    static bool decode(const Node &node, cv::Ptr<cv::xfeatures2d::SIFT> &rhs) {
        namespace cfg = descry::config::features;

        if (!node.IsMap())
            return false;

        // optionals
        auto nFeatures = node[cfg::RETAIN_FEATURES].as<int>(0);
        auto nOctaveLayers = node[cfg::OCTAVE_LAYERS].as<int>(3);
        auto contrastThreshold = node[cfg::CONTRAST_THRESH].as<double>(0.04);
        auto edgeThreshold = node[cfg::EDGE_THRESH].as<double>(10.);
        auto sigma = node[cfg::SIGMA].as<double>(1.6);

        rhs = cv::xfeatures2d::SIFT::create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

        return true;
    }
};

}

namespace descry {

namespace {

template <class Descriptor>
struct pcl_describer_parser {};

template <>
struct pcl_describer_parser<pcl::SHOT352> {
    using describer_t = SHOT_PCL;

    static bool is_config_matching(const std::string& type_str) {
        return type_str == config::features::SHOT_PCL_TYPE;
    }

    static void add_rfs_if_supported(describer_t& describer, const DualRefFrames& rfs) {
        describer.setInputReferenceFrames(rfs.host());
    }
};

template <>
struct pcl_describer_parser<pcl::FPFHSignature33> {
    using describer_t = FPFH_PCL;

    static bool is_config_matching(const std::string& type_str) {
        return type_str == config::features::FPFH_PCL_TYPE;
    }

    static void add_rfs_if_supported(describer_t& /*describer*/, const DualRefFrames& /*rfs*/) {}
};

}

template<class D>
Description<D> Describer<D>::compute(const Image& image) {
    if (!_descr)
        DESCRY_THROW(NotConfiguredException, "Describer not configured");
    auto description =_descr(image);
    viewer_.show(image, description.getKeypoints());
    return description;
}

template<class D>
bool Describer<D>::configure(const Config& cfg) {
    auto keys_det = KeypointDetector{};
    auto& keys_cfg = cfg[config::keypoints::NODE_NAME];
    if (!keys_cfg || !keys_det.configure(keys_cfg)) // required
        return false;

    auto rf_est = RefFramesEstimation{};
    auto& rf_cfg = cfg[config::ref_frames::NODE_NAME];
    if (rf_cfg && !rf_est.configure(rf_cfg)) // optional
        return false;

    auto features_config = cfg[config::features::NODE_NAME];
    if (!features_config)
        DESCRY_THROW(InvalidConfigException, "missing features node");

    if (!features_config[config::TYPE_NODE])
        return false;
    auto est_type = features_config[config::TYPE_NODE].as<std::string>();

    try {
        if (pcl_describer_parser<D>::is_config_matching(est_type)) {
            auto descr = features_config.as<typename pcl_describer_parser<D>::describer_t>();
            _descr = [ descr{std::move(descr)}, keys_det{std::move(keys_det)}, rf_est{std::move(rf_est)}, this ]
            (const Image &image) mutable {
                auto d = Description<D>();
                auto latency = measure_latency(config::keypoints::NODE_NAME, this->log_latency_);
                d.setKeypoints(keys_det.compute(image));
                latency.finish();
                descr.setInputCloud(d.getKeypoints().getShape().host());
                descr.setSearchSurface(image.getShapeCloud().host());
                descr.setInputNormals(image.getNormals().host());
                if (rf_est.is_configured()) {
                    latency.start(config::ref_frames::NODE_NAME);
                    d.setRefFrames(rf_est.compute(image, d.getKeypoints()));
                    latency.finish();
                    pcl_describer_parser<D>::add_rfs_if_supported(descr, d.getRefFrames());
                }
                auto features = make_cloud<D>();
                latency.start(config::features::NODE_NAME);
                descr.compute(*features);
                latency.finish();
                d.setFeatures(cupcl::DualContainer<D>{features});
                return d;
            };
        } else
            return false;
    } catch ( const YAML::BadConversion& e) {
        return false;
    }

    viewer_.configure(cfg);
    if (cfg[config::LOG_LATENCY])
        log_latency_ = cfg[config::LOG_LATENCY].as<bool>();

    return true;
}

namespace {

bool is_finite_keypoint(const cv::KeyPoint& key, const Image& image) {
    auto x = static_cast<unsigned int>(key.pt.x);
    auto y = static_cast<unsigned int>(key.pt.y);

    const auto& shape = image.getShapeCloud().host();

    return pcl::isFinite(shape->at(x, y));
}

struct ColorDescription {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

ColorDescription filter_null_keypoints(const ColorDescription& descr, const Image& image) {
    auto filtered = ColorDescription{};

    for (std::size_t idx = 0; idx < descr.keypoints.size(); ++idx) {
        if (is_finite_keypoint(descr.keypoints.at(idx), image)) {
            filtered.keypoints.emplace_back(descr.keypoints.at(idx));
            filtered.descriptors.push_back(descr.descriptors.row(idx));
        }
    }

    return filtered;
}

}

template<>
bool Describer<cv::Mat>::configure(const Config& cfg) {
    auto features_config = cfg[config::features::NODE_NAME];
    if (!features_config)
        DESCRY_THROW(InvalidConfigException, "missing features node");

    if (!features_config[config::TYPE_NODE])
        return false;
    auto est_type = features_config[config::TYPE_NODE].as<std::string>();
    cv::Ptr<cv::Feature2D> descr{};
    try {
        if (est_type == config::features::ORB_TYPE)
            descr = features_config.as<cv::Ptr<cv::ORB>>();
        else if (est_type == config::features::AKAZE_TYPE)
            descr = features_config.as<cv::Ptr<cv::AKAZE>>();
        else if (est_type == config::features::SIFT_TYPE)
            descr = features_config.as<cv::Ptr<cv::xfeatures2d::SIFT>>();
        else
            return false;
    } catch ( const YAML::BadConversion& e) {
        return false;
    }

    _descr = [ descr{std::move(descr)}, this ] (const Image& image) mutable {
        auto color = ColorDescription{};
        {
            auto scoped_latency = measure_scope_latency(config::features::NODE_NAME, this->log_latency_);
            descr->detectAndCompute(image.getColorMat(), cv::noArray(), color.keypoints, color.descriptors);
        }
        auto filtered_color = filter_null_keypoints(color, image);
        auto d = Description<cv::Mat>{};
        d.setKeypoints(Keypoints{std::move(filtered_color.keypoints), image});
        d.setFeatures(std::move(filtered_color.descriptors));
        return d;
    };

    viewer_.configure(cfg);
    if (cfg[config::LOG_LATENCY])
        log_latency_ = cfg[config::LOG_LATENCY].as<bool>();

    return true;
}

template
class Describer<pcl::SHOT352>;

template
class Describer<pcl::FPFHSignature33>;

template
class Describer<cv::Mat>;

}
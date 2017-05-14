#include <descry/descriptors.h>

#include <opencv2/features2d.hpp>

#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>

using SHOT_PCL = pcl::SHOTEstimationOMP<descry::ShapePoint, pcl::Normal, pcl::SHOT352>;
using FPFH_PCL = pcl::FPFHEstimationOMP<descry::ShapePoint, pcl::Normal, pcl::FPFHSignature33>;

namespace YAML {

template <class DescriberPCL>
bool decode_pcl_omp_describer(const YAML::Node& node, DescriberPCL& rhs)
{
    if (!node.IsMap())
        return false;

    if (node[descry::config::descriptors::SUPPORT_RAD])
        rhs.setRadiusSearch(node[descry::config::descriptors::SUPPORT_RAD].as<double>());
    else
        return false;

    if (node[descry::config::descriptors::THREADS])
        rhs.setNumberOfThreads(node[descry::config::descriptors::THREADS].as<unsigned>());

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
        namespace cfg = descry::config::descriptors;

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

}

namespace descry {

namespace {

template <class Descriptor>
struct pcl_describer_parser {};

template <>
struct pcl_describer_parser<pcl::SHOT352> {
    using describer_t = SHOT_PCL;

    static bool is_config_matching(const std::string& type_str) {
        return type_str == config::descriptors::SHOT_PCL_TYPE;
    }

    static void add_rfs_if_supported(describer_t& descr, const Image& image) {
        descr.setInputReferenceFrames(image.getRefFrames().host());
    }
};

template <>
struct pcl_describer_parser<pcl::FPFHSignature33> {
    using describer_t = FPFH_PCL;

    static bool is_config_matching(const std::string& type_str) {
        return type_str == config::descriptors::FPFH_PCL_TYPE;
    }

    static void add_rfs_if_supported(describer_t&, const Image&) {}
};

}

template<class D>
DescriptorContainer<D> Describer<D>::compute(const Image& image) {
    if (!_descr)
        DESCRY_THROW(NotConfiguredException, "Describer not configured");
    return _descr(image);
}

template<class D>
bool Describer<D>::configure(const Config& config) {
    if (!config["type"])
        return false;

    auto est_type = config["type"].as<std::string>();
    try {
        if (pcl_describer_parser<D>::is_config_matching(est_type)) {
            auto descr = config.as<typename pcl_describer_parser<D>::describer_t>();
            _descr = [ descr{std::move(descr)} ] (const Image &image) mutable {
                descr.setInputCloud(image.getKeypoints().getShape().host());
                descr.setSearchSurface(image.getShapeCloud().host());
                descr.setInputNormals(image.getNormals().host());
                pcl_describer_parser<D>::add_rfs_if_supported(descr, image);
                typename pcl::PointCloud<D>::Ptr features{new pcl::PointCloud<D>{}};
                descr.compute(*features);
                return cupcl::DualContainer<D>{features};
            };
        } else
            return false;
    } catch ( const YAML::BadConversion& e) {
        return false;
    }

    return true;
}

namespace {

bool is_finite_keypoint(const cv::KeyPoint& key, const Image& image) {
    auto x = static_cast<unsigned int>(key.pt.x);
    auto y = static_cast<unsigned int>(key.pt.y);

    const auto& shape = image.getShapeCloud().host();

    return pcl::isFinite(shape->at(x, y));
}

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
bool Describer<ColorDescription>::configure(const Config& config) {
    if (!config["type"])
        return false;

    auto est_type = config["type"].as<std::string>();
    try {
        if (est_type == config::descriptors::ORB_TYPE) {
            auto descr = config.as<cv::Ptr<cv::ORB>>();
            _descr = [ descr{std::move(descr)} ] (const Image& image) mutable {
                auto d = ColorDescription{};
                descr->detectAndCompute(image.getColorMat(), cv::noArray(), d.keypoints, d.descriptors);
                return filter_null_keypoints(d, image);
            };
        } else
            return false;
    } catch ( const YAML::BadConversion& e) {
        return false;
    }

    return true;
}

template
class Describer<pcl::SHOT352>;

template
class Describer<pcl::FPFHSignature33>;

template
class Describer<ColorDescription>;

}
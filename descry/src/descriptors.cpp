#include <descry/descriptors.h>

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
cupcl::DualContainer<D> Describer<D>::compute(const Image& image) {
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

template
class Describer<pcl::SHOT352>;

template
class Describer<pcl::FPFHSignature33>;

}
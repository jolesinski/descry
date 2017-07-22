#include <descry/normals.h>
#include <descry/cupcl/normals.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <descry/config/common.h>
#include <descry/latency.h>

using NEstOMP = pcl::NormalEstimationOMP<descry::FullPoint, pcl::Normal>;
using NEstINT = pcl::IntegralImageNormalEstimation<descry::FullPoint, pcl::Normal>;

namespace YAML {
template<>
struct convert<NEstOMP> {
    static bool decode(const Node& node, NEstOMP& rhs) {
        if(!node.IsMap()) {
            return false;
        }

        if (node[descry::config::normals::SUPPORT_RAD])
            rhs.setRadiusSearch(node[descry::config::normals::SUPPORT_RAD].as<double>());
        else
            return false;

        if (node[descry::config::normals::THREADS])
            rhs.setNumberOfThreads(node[descry::config::normals::THREADS].as<unsigned>());

        return true;
    }
};

template<>
struct convert<NEstINT> {
    static bool decode(const Node& node, NEstINT& rhs) {
        if(!node.IsMap()) {
            return false;
        }

        {
            auto elem = node[descry::config::normals::MAX_DEPTH_CHANGE];
            if (elem)
                rhs.setMaxDepthChangeFactor(elem.as<float>());
        }

        {
            auto elem = node[descry::config::normals::SMOOTHING_SIZE];
            if (elem)
                rhs.setNormalSmoothingSize(elem.as<float>());
        }

        {
            auto elem = node[descry::config::normals::INTEGRAL_METHOD];
            if (elem) {
                auto method_str = elem.as<std::string>();
                if (method_str == "covariance")
                    rhs.setNormalEstimationMethod(rhs.COVARIANCE_MATRIX);
                else if (method_str == "average_gradient")
                    rhs.setNormalEstimationMethod(rhs.AVERAGE_3D_GRADIENT);
                else if (method_str == "average_depth_change")
                    rhs.setNormalEstimationMethod(rhs.AVERAGE_DEPTH_CHANGE);
                else
                    return false;
            }
        }

        return true;
    }
};
}

namespace descry {

template<class NEst>
auto configureEstimatorPCL(const YAML::Node& node) {
    auto nest = node.as<NEst>();
    return [ nest{std::move(nest)} ]
            (const Image &image) mutable {
                nest.setInputCloud(image.getFullCloud().host());
                Normals::Ptr normals{new Normals{}};
                nest.compute(*normals);
                return DualNormals{normals};
            };
}

bool NormalEstimation::configure(const Config& cfg) {
    if (!cfg["type"])
        return false;

    auto est_type = cfg["type"].as<std::string>();
    if (est_type == config::normals::OMP_TYPE) {
        nest_ = configureEstimatorPCL<NEstOMP>(cfg);
    } else if (est_type == config::normals::INTEGRAL_IMAGE_TYPE) {
        nest_ = configureEstimatorPCL<NEstINT>(cfg);
    } else if (est_type == config::normals::CUPCL_TYPE && cfg[config::normals::SUPPORT_RAD]) {
        auto rad = cfg[config::normals::SUPPORT_RAD].as<float>();
        nest_ = [rad](const Image &image) {
            return cupcl::computeNormals(image.getShapeCloud(), image.getProjection(), rad);
        };
    } else
        return false;

    viewer_.configure(cfg);
    log_latency_ = cfg[config::LOG_LATENCY].as<bool>(false);
    std::cerr << "Log latency" << log_latency_ << std::endl;
    return true;
}

DualNormals NormalEstimation::compute(const Image& image) const {
    if (!nest_)
        DESCRY_THROW(NotConfiguredException, "Normals not configured");

    auto latency = descry::measure_latency("Normals", log_latency_);
    auto normals = nest_(image);
    latency.finish();

    viewer_.show(image.getFullCloud().host(), normals.host());

    return normals;
}

}
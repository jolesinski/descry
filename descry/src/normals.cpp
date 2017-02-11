#include <descry/normals.h>
#include <descry/cupcl/normals.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

using NEstOMP = pcl::NormalEstimationOMP<descry::FullPoint, pcl::Normal>;
using NEstINT = pcl::IntegralImageNormalEstimation<descry::FullPoint, pcl::Normal>;

namespace YAML {
template<>
struct convert<NEstOMP> {
    static bool decode(const Node& node, NEstOMP& rhs) {
        if(!node.IsMap()) {
            return false;
        }

        if (node["k-support"])
            rhs.setKSearch(node["k-support"].as<int>());
        else if (node["r-support"])
            rhs.setRadiusSearch(node["r-support"].as<double>());
        else
            false;

        if (node["threads"])
            rhs.setNumberOfThreads(node["threads"].as<unsigned>());

        return true;
    }
};

template<>
struct convert<NEstINT> {
    static bool decode(const Node& node, NEstINT& rhs) {
        if(!node.IsMap()) {
            return false;
        }

        auto elem = node["max-depth-change"];
        if (elem)
            rhs.setMaxDepthChangeFactor(elem.as<float>());

        elem = node["smoothing"];
        if (elem)
            rhs.setNormalSmoothingSize(elem.as<float>());

        elem = node["method"];
        if (elem)
        {
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
                descry::Normals::Ptr normals{new descry::Normals{}};
                nest.compute(*normals);
                return normals;
            };
}

bool NormalEstimation::configure(const Config& config) {
    if (!config["type"])
        return false;

    auto est_type = config["type"].as<std::string>();
    if (est_type == "omp") {
        _nest = configureEstimatorPCL<NEstOMP>(config);
    } else if (est_type == "int") {
        _nest = configureEstimatorPCL<NEstINT>(config);
    } else if (est_type == "cupcl" && config["r-support"]) {
        auto rad = config["r-support"].as<float>();
        _nest = [rad](const Image &image) {
            auto normals = cupcl::computeNormals(image.getShapeCloud(),
                                                 image.getProjection(),
                                                 rad);
            return normals.host();
        };
    } else
        return false;

    return true;
}

Normals::Ptr NormalEstimation::compute(const Image& image) {
    assert(_nest);
    return _nest(image);
}

}
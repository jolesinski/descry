
#include <descry/normals.h>

#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

using NEstOMP = pcl::NormalEstimationOMP<descry::Point, pcl::Normal>;
using NEstINT = pcl::IntegralImageNormalEstimation<descry::Point, pcl::Normal>;

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

        if (node["threads"])
            rhs.setNumberOfThreads(node["threads"].as<int>());

        return true;
    }
};
}

bool descry::NormalEstimation::configure(const descry::Config& config) {
    if (!config["type"])
        return false;

    auto est_type = config["type"].as<std::string>();
    if (est_type == "omp") {
        auto nest_omp = config.as<NEstOMP>();
        _nest = [ nest{std::move(nest_omp)} ]
                (const auto& cloud) mutable
                {
                    nest.setInputCloud(cloud);
                    descry::Normals::Ptr normals(new descry::Normals{});
                    nest.compute(*normals);
                    return normals;
                };
    } else if (est_type == "integral") {
        auto nest_int = std::make_unique<NEstINT>();
        if (config["r-support"])
            nest_int->setRadiusSearch(config["r-support"].as<double>());
    } else
        return false;

    return true;
}

descry::Normals::Ptr
descry::NormalEstimation::compute(const descry::PointCloud::ConstPtr& cloud) {
    assert(_nest);
    return _nest(cloud);
}

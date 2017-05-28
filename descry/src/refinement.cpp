#include <descry/refinement.h>

#include <pcl/registration/icp_nl.h>

namespace descry {
    using BasicICP = pcl::IterativeClosestPoint<descry::ShapePoint, descry::ShapePoint>;
}

namespace YAML {
template<>
struct convert<descry::BasicICP> {
    static bool decode(const Node& node, descry::BasicICP& rhs) {
        using namespace descry::config::refiner;

        if (!node.IsMap())
            return false;

        // optionals
        {
            auto &elem = node[MAX_ITERATIONS];
            if (elem)
                rhs.setMaximumIterations(elem.as<int>());
        }
        {
            auto &elem = node[MAX_CORRESPONDENCE_DISTANCE];
            if (elem)
                rhs.setMaxCorrespondenceDistance(elem.as<double>());
        }
        {
            auto &elem = node[TRANSFORMATION_EPSILON];
            if (elem)
                rhs.setTransformationEpsilon(elem.as<double>());
        }
        {
            auto &elem = node[EUCLIDEAN_FITNESS_THRESH];
            if (elem)
                rhs.setEuclideanFitnessEpsilon(elem.as<double>());
        }
        {
            auto &elem = node[USE_RECIPROCAL];
            if (elem)
                rhs.setUseReciprocalCorrespondences(elem.as<bool>());
        }

        return true;
    }
};
}

namespace descry {

void Refiner::configure(const Config& config) {
    if (!config[config::TYPE_NODE])
        DESCRY_THROW(InvalidConfigException, "missing type config");

    auto type_name = config[config::TYPE_NODE].as<std::string>();

    try {
        if (type_name == config::refiner::ICP_TYPE) {
            auto icp = config.as<BasicICP>();
            model_feed_ = [ icp{std::move(icp)} ] (const Model &model) mutable {
                auto shape_model = descry::make_cloud<descry::ShapePoint>();
                pcl::copyPointCloud(*model.getFullCloud(), *shape_model);

                icp.setInputSource(shape_model);

                return [icp{std::move(icp)}] (const Image& scene, const Instances& instances) mutable {
                    auto refined_instances = Instances{ instances.cloud, {} };
                    icp.setInputTarget(scene.getShapeCloud().host());

                    for (auto pose : instances.poses) {
                        auto refined = descry::make_cloud<descry::ShapePoint>();
                        icp.align(*refined, pose);
                        if (icp.hasConverged())
                            refined_instances.poses.emplace_back(icp.getFinalTransformation());
                    }
                    return refined_instances;
                };
            };
        } else
            DESCRY_THROW(InvalidConfigException, "unsupported refiner type");
    } catch ( const YAML::BadConversion& e) {
        DESCRY_THROW(InvalidConfigException, "yaml conversion failed");
    }
}

void Refiner::train(const Model& model) {
    if (!model_feed_)
        DESCRY_THROW(NotConfiguredException, "Refinement not configured");
    refiner_ = model_feed_(model);
}

Instances Refiner::compute(const Image& scene, const Instances& instances) {
    if (!refiner_)
        DESCRY_THROW(NotConfiguredException, "Refinement not configured");
    return refiner_(scene, instances);
}

}

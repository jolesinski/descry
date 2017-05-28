#include <descry/verification.h>

// speedup
#define PCL_NO_PRECOMPILE
#include <pcl/recognition/hv/hv_go.h>
#include <pcl/recognition/hv/hv_papazov.h>

namespace descry {
using HVPapazov = pcl::PapazovHV<descry::ShapePoint, descry::ShapePoint>;
using HVGlobal = pcl::GlobalHypothesesVerification<descry::ShapePoint, descry::ShapePoint>;
}

namespace YAML {
template<>
struct convert<descry::HVPapazov> {
    static bool decode(const Node& node, descry::HVPapazov& rhs) {
        using namespace descry::config::verifier;

        if (!node.IsMap())
            return false;

        // optionals
        {
            auto &elem = node[RESOLUTION];
            if (elem)
                rhs.setResolution(elem.as<float>());
        }
        {
            auto &elem = node[INLIER_THRESH];
            if (elem)
                rhs.setInlierThreshold(elem.as<float>());
        }
        {
            auto &elem = node[SUPPORT_THRESH];
            if (elem)
                rhs.setSupportThreshold(elem.as<float>());
        }
        {
            auto &elem = node[PENALTY_THRESH];
            if (elem)
                rhs.setPenaltyThreshold(elem.as<float>());
        }
        {
            auto &elem = node[CONFLICT_THRESH];
            if (elem)
                rhs.setConflictThreshold(elem.as<float>());
        }

        return true;
    }
};
}

namespace descry {

namespace {

std::vector<descry::ShapeCloud::ConstPtr> instantiate(const Instances& instances) {
    auto aligned_models = std::vector<descry::ShapeCloud::ConstPtr>{};
    auto shape_model = descry::make_cloud<descry::ShapePoint>();
    pcl::copyPointCloud(*instances.cloud, *shape_model);
    for (auto pose : instances.poses) {
        auto transformed = descry::make_cloud<descry::ShapePoint>();
        pcl::transformPointCloud(*shape_model, *transformed, pose);
        aligned_models.emplace_back(transformed);
    }

    return aligned_models;
}

}

void Verifier::configure(const Config& config) {
    if (!config[config::TYPE_NODE])
        DESCRY_THROW(InvalidConfigException, "missing type config");

    auto type_name = config[config::TYPE_NODE].as<std::string>();


    try {
        if (type_name == config::verifier::PAPAZOV_TYPE) {
            auto hv = config.as<HVPapazov>();
            verifier_ = [ hv{std::move(hv)} ] (const Image &image, const Instances& instances) mutable {
                hv.setSceneCloud(image.getShapeCloud().host());

                auto aligned_models = instantiate(instances);
                hv.addModels(aligned_models, true);

                hv.verify();
                auto mask = std::vector<bool>{};
                hv.getMask(mask);

                auto verified_instances = Instances{ instances.cloud, {} };
                for (auto idx = 0u; idx < mask.size(); ++idx) {
                    if (mask[idx])
                        verified_instances.poses.emplace_back(instances.poses.at(idx));
                }

                return verified_instances;
            };
        } else
            DESCRY_THROW(InvalidConfigException, "unsupported refiner type");
    } catch ( const YAML::BadConversion& e) {
        DESCRY_THROW(InvalidConfigException, "yaml conversion failed");
    }
}

Instances Verifier::compute(const Image& scene, const Instances& instances) {
    if (!verifier_)
        DESCRY_THROW(NotConfiguredException, "Verifier not configured");
    return verifier_(scene, instances);
}

}
#include <descry/keypoints.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/keypoints/iss_3d.h>
#include <descry/cupcl/iss.h>

using namespace descry::config::keypoints;

using KDetUniform = pcl::UniformSampling<descry::ShapePoint>;
using KDetISS = pcl::ISSKeypoint3D<descry::ShapePoint, descry::ShapePoint>;

namespace YAML {
template<>
struct convert<KDetUniform> {
    static bool decode(const Node& node, KDetUniform& rhs) {
        if(!node.IsMap())
            return false;

        // required
        if (!node[SUPPORT_RAD])
            return false;

        rhs.setRadiusSearch(node[SUPPORT_RAD].as<double>());
        return true;
    }
};

template<>
struct convert<KDetISS> {
    static bool decode(const Node& node, KDetISS& rhs) {
        if(!node.IsMap())
            return false;

        // required
        if (!node[SALIENT_RAD] || !node[NON_MAX_RAD])
            return false;

        rhs.setSalientRadius(node[SALIENT_RAD].as<double>());
        rhs.setNonMaxRadius(node[NON_MAX_RAD].as<double>());

        // optionals
        {
            auto &elem = node[BORDER_RAD];
            if (elem)
                rhs.setBorderRadius(elem.as<double>());
        }

        {
            auto &elem = node[NORMAL_RAD];
            if (elem)
                rhs.setNormalRadius(elem.as<double>());
        }

        {
            auto &elem = node[LAMBDA_RATIO_21];
            if (elem)
                rhs.setThreshold21(elem.as<double>());
        }

        {
            auto &elem = node[LAMBDA_RATIO_32];
            if (elem)
                rhs.setThreshold32(elem.as<double>());
        }

        {
            auto &elem = node[BOUNDARY_ANGLE];
            if (elem)
                rhs.setAngleThreshold(elem.as<float>());
        }

        {
            auto &elem = node[MIN_NEIGHBOURS];
            if (elem)
                rhs.setMinNeighbors(elem.as<int>());
        }

        {
            auto &elem = node[THREADS];
            if (elem)
                rhs.setNumberOfThreads(elem.as<unsigned>());
        }

        return true;
    }
};

template<>
struct convert<descry::cupcl::ISSConfig> {
    static bool decode(const Node& node, descry::cupcl::ISSConfig& rhs) {
        if(!node.IsMap())
            return false;

        // required
        if (!node[SALIENT_RAD] || !node[NON_MAX_RAD])
            return false;

        rhs.salient_rad = node[SALIENT_RAD].as<float>();
        rhs.non_max_rad = node[NON_MAX_RAD].as<float>();

        {
            auto &elem = node[descry::config::keypoints::LAMBDA_RATIO_21];
            if (elem)
                rhs.lambda_ratio_21 = elem.as<float>();
        }

        {
            auto &elem = node[descry::config::keypoints::LAMBDA_RATIO_32];
            if (elem)
                rhs.lambda_ratio_32 = elem.as<float>();
        }

        {
            auto &elem = node[descry::config::keypoints::LAMBDA_THRESHOLD_3];
            if (elem)
                rhs.lambda_threshold_3 = elem.as<float>();
        }

        {
            auto &elem = node[descry::config::keypoints::MIN_NEIGHBOURS];
            if (elem)
                rhs.min_neighs = elem.as<unsigned int>();
        }

        return true;
    }
};
}

namespace descry {

bool ShapeKeypointDetector::configure(const Config& config) {
    if (!config["type"])
        return false;

    auto est_type = config["type"].as<std::string>();

    try {
        if (est_type == config::keypoints::UNIFORM_TYPE) {
            auto nest = config.as<KDetUniform>();
            _nest = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                ShapeCloud::Ptr keypoints{new ShapeCloud{}};
                nest.filter(*keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == config::keypoints::ISS_TYPE) {
            auto nest = config.as<KDetISS>();
            _nest = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                if (!image.getNormals().empty())
                    nest.setNormals(image.getNormals().host());
                ShapeCloud::Ptr keypoints{new ShapeCloud{}};
                nest.compute(*keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == config::keypoints::ISS_CUPCL_TYPE) {
            _nest = [ iss_cfg{config.as<cupcl::ISSConfig>()} ] (const Image &image) {
                return cupcl::computeISS(image.getShapeCloud(), image.getProjection(), iss_cfg);
            };
        } else
            return false;
    } catch ( const YAML::BadConversion& e) {
        return false;
    }

    return true;
}

ShapeKeypoints ShapeKeypointDetector::compute(const Image& image) {
    assert(_nest);
    return _nest(image);
}

}
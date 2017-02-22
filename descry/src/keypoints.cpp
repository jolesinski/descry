#include <descry/keypoints.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/keypoints/iss_3d.h>
#include <descry/cupcl/iss.h>


using KDetUniform = pcl::UniformSampling<descry::ShapePoint>;
using KDetISS = pcl::ISSKeypoint3D<descry::ShapePoint, descry::ShapePoint>;

namespace YAML {
template<>
struct convert<KDetUniform> {
    static bool decode(const Node& node, KDetUniform& rhs) {
        if(!node.IsMap())
            return false;

        if (!node["r-support"])
            return false;

        rhs.setRadiusSearch(node["r-support"].as<double>());
        return true;
    }
};

template<>
struct convert<KDetISS> {
    static bool decode(const Node& node, KDetISS& rhs) {
        if(!node.IsMap())
            return false;

        // required
        if (!node["salient-radius"] || !node["non-max-radius"])
            return false;

        rhs.setSalientRadius(node["salient-radius"].as<double>());
        rhs.setNonMaxRadius(node["non-max-radius"].as<double>());

        {
            auto &elem = node["border-radius"];
            if (elem)
                rhs.setBorderRadius(elem.as<double>());
        }

        {
            auto &elem = node["normal-radius"];
            if (elem)
                rhs.setNormalRadius(elem.as<double>());
        }

        {
            auto &elem = node["threshold-21"];
            if (elem)
                rhs.setThreshold21(elem.as<double>());
        }

        {
            auto &elem = node["threshold-32"];
            if (elem)
                rhs.setThreshold32(elem.as<double>());
        }

        {
            auto &elem = node["threshold-angle"];
            if (elem)
                rhs.setAngleThreshold(elem.as<float>());
        }

        {
            auto &elem = node["min-neighbours"];
            if (elem)
                rhs.setMinNeighbors(elem.as<int>());
        }

        {
            auto &elem = node["threads"];
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
        if (!node["salient-radius"] || !node["non-max-radius"])
            return false;

        rhs.salient_rad = node["salient-radius"].as<float>();
        rhs.non_max_rad = node["non-max-radius"].as<float>();

        {
            auto &elem = node["lambda-ratio-21"];
            if (elem)
                rhs.lambda_ratio_21 = elem.as<float>();
        }

        {
            auto &elem = node["lambda-ratio-32"];
            if (elem)
                rhs.lambda_ratio_32 = elem.as<float>();
        }

        {
            auto &elem = node["lambda-threshold-3"];
            if (elem)
                rhs.lambda_threshold_3 = elem.as<float>();
        }

        {
            auto &elem = node["min-neighbours"];
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
        if (est_type == "uniform") {
            auto nest = config.as<KDetUniform>();
            _nest = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                ShapeCloud::Ptr keypoints{new ShapeCloud{}};
                nest.filter(*keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == "iss") {
            auto nest = config.as<KDetISS>();
            _nest = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                if (!image.getNormals().empty())
                    nest.setNormals(image.getNormals().host());
                ShapeCloud::Ptr keypoints{new ShapeCloud{}};
                nest.compute(*keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == "iss-cupcl") {
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
#include <descry/keypoints.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/keypoints/iss_3d.h>

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
}

namespace descry {

template<class KDet>
auto configureDetectorPCL(const YAML::Node& node) {
    auto nest = node.as<KDet>();
    return [ nest{std::move(nest)} ]
    (const Image &image) mutable {
        nest.setInputCloud(image.getFullCloud().host());
        descry::Normals::Ptr normals{new descry::Normals{}};
        nest.compute(*normals);
        return DualNormals{normals};
    };
}

bool ShapeKeypointDetector::configure(const Config& config) {
    if (!config["type"])
        return false;

    auto est_type = config["type"].as<std::string>();

    try {
        if (est_type == "uniform") {
            auto nest = config.as<KDetUniform>();
            _nest = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                ShapeKeypoints::Ptr keypoints{new ShapeKeypoints{}};
                nest.filter(*keypoints);
                return keypoints;
            };
        } else if (est_type == "iss") {
            auto nest = config.as<KDetISS>();
            _nest = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                if (!image.getNormals().empty())
                    nest.setNormals(image.getNormals().host());
                ShapeKeypoints::Ptr keypoints{new ShapeKeypoints{}};
                nest.compute(*keypoints);
                return keypoints;
            };
        } else
            return false;
    } catch ( const YAML::BadConversion& e) {
        return false;
    }

    return true;
}

ShapeKeypoints::Ptr ShapeKeypointDetector::compute(const Image& image) {
    assert(_nest);
    return _nest(image);
}

}
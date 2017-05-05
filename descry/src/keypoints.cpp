#include <descry/keypoints.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/susan.h>
#include <descry/cupcl/iss.h>

using namespace descry::config::keypoints;

using KDetUniform = pcl::UniformSampling<pcl::PointXYZ>;
using KDetISS = pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ>;
using KDetHarris = pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>;
using KDetSIFT = pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointXYZI>;
using KDetSUSAN = pcl::SUSANKeypoint<pcl::PointXYZ, pcl::PointXYZ>;

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
struct convert<KDetHarris::ResponseMethod> {
    static bool decode(const Node& node, KDetHarris::ResponseMethod& rhs) {
        //typedef enum {HARRIS = 1, NOBLE, LOWE, TOMASI, CURVATURE} ResponseMethod;
        auto method_name = node.as<std::string>();

        if (method_name == METHOD_HARRIS)
            rhs = KDetHarris::ResponseMethod::HARRIS;
        else if (method_name == METHOD_NOBLE)
            rhs = KDetHarris::ResponseMethod::NOBLE;
        else if (method_name == METHOD_LOWE)
            rhs = KDetHarris::ResponseMethod::LOWE;
        else if (method_name == METHOD_TOMASI)
            rhs = KDetHarris::ResponseMethod::TOMASI;
        else if (method_name == METHOD_CURVATURE)
            rhs = KDetHarris::ResponseMethod::CURVATURE;
        else
            return false;

        return true;
    }
};

template<>
struct convert<KDetHarris> {
    static bool decode(const Node& node, KDetHarris& rhs) {
        if(!node.IsMap())
            return false;

        // required
        if (!node[SUPPORT_RAD])
            return false;

        rhs.setRadius(node[SUPPORT_RAD].as<double>());

        // optionals
        {
            auto &elem = node[USE_NONMAX];
            if (elem)
                rhs.setNonMaxSupression(elem.as<bool>());
        }

        {
            auto &elem = node[HARRIS_THRESHOLD];
            if (elem)
                rhs.setThreshold(elem.as<float>());
        }

        {
            auto &elem = node[USE_REFINE];
            if (elem)
                rhs.setRefine(elem.as<bool>());
        }

        {
            auto &elem = node[METHOD_NAME];
            if (elem)
                rhs.setMethod(elem.as<KDetHarris::ResponseMethod>());
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
struct convert<KDetSIFT> {
    static bool decode(const Node& node, KDetSIFT& rhs) {
        if(!node.IsMap())
            return false;

        // required
        if (!node[MIN_SCALE] || !node[OCTAVES] || !node[SCALES_PER_OCTAVE])
            return false;

        rhs.setScales(node[MIN_SCALE].as<float>(), node[OCTAVES].as<int>(), node[SCALES_PER_OCTAVE].as<int>());

        // optionals
        {
            auto &elem = node[MIN_CONTRAST];
            if (elem)
                rhs.setMinimumContrast(elem.as<float>());
        }

        return true;
    }
};

template<>
struct convert<KDetSUSAN> {
    static bool decode(const Node& node, KDetSUSAN& rhs) {
        if(!node.IsMap())
            return false;

        // required
        if (!node[SUPPORT_RAD])
            return false;

        rhs.setRadius(node[SUPPORT_RAD].as<float>());

        // optionals
        {
            auto &elem = node[USE_NONMAX];
            if (elem)
                rhs.setNonMaxSupression(elem.as<bool>());
        }

        {
            auto &elem = node[ANGULAR_THRESH];
            if (elem)
                rhs.setAngularThreshold(elem.as<float>());
        }

        {
            auto &elem = node[DISTANCE_THRESH];
            if (elem)
                rhs.setDistanceThreshold(elem.as<float>());
        }

        {
            auto &elem = node[INTENSITY_THRESH];
            if (elem)
                rhs.setIntensityThreshold(elem.as<float>());
        }

        {
            auto &elem = node[USE_VALIDATION];
            if (elem)
                rhs.setGeometricValidation(elem.as<bool>());
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
            nest_ = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                auto keypoints = make_cloud<pcl::PointXYZ>();
                nest.filter(*keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == config::keypoints::ISS_TYPE) {
            auto nest = config.as<KDetISS>();
            nest_ = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                if (!image.getNormals().empty())
                    nest.setNormals(image.getNormals().host());
                auto keypoints = make_cloud<pcl::PointXYZ>();
                nest.compute(*keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == config::keypoints::HARRIS_TYPE) {
            auto nest = config.as<KDetHarris>();
            nest_ = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                if (!image.getNormals().empty())
                    nest.setNormals(image.getNormals().host());
                auto intensity_keys = make_cloud<pcl::PointXYZI>();
                nest.compute(*intensity_keys);
                auto keypoints = make_cloud<pcl::PointXYZ>();
                pcl::copyPointCloud(*intensity_keys, *keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == config::keypoints::SIFT_PCL_TYPE) {
            auto nest = config.as<KDetSIFT>();
            nest_ = [ nest{std::move(nest)} ] (const Image &image) mutable {
                auto point_normals = make_cloud<pcl::PointNormal>();
                pcl::concatenateFields(*image.getShapeCloud().host(), *image.getNormals().host(), *point_normals);
                nest.setInputCloud(point_normals);
                auto intensity_keys = make_cloud<pcl::PointXYZI>();
                nest.compute(*intensity_keys);
                auto keypoints = make_cloud<pcl::PointXYZ>();
                pcl::copyPointCloud(*intensity_keys, *keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == config::keypoints::SUSAN_PCL_TYPE) {
            auto nest = config.as<KDetSUSAN>();
            nest_ = [ nest{std::move(nest)} ] (const Image &image) mutable {
                nest.setInputCloud(image.getShapeCloud().host());
                if (!image.getNormals().empty())
                    nest.setNormals(image.getNormals().host());
                auto keypoints = make_cloud<pcl::PointXYZ>();
                nest.compute(*keypoints);
                return ShapeKeypoints{keypoints};
            };
        } else if (est_type == config::keypoints::ISS_CUPCL_TYPE) {
            nest_ = [ iss_cfg{config.as<cupcl::ISSConfig>()} ] (const Image &image) {
                return cupcl::computeISS(image.getShapeCloud(), image.getProjection(), iss_cfg);
            };
        } else
            return false;
    } catch ( const YAML::BadConversion& e) {
        return false;
    }

    return true;
}

ShapeKeypoints ShapeKeypointDetector::compute(const Image& image) const {
    if (!nest_)
        DESCRY_THROW(NotConfiguredException, "Keypoints not configured");
    return nest_(image);
}

}
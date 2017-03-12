#include <descry/projector.h>

#include <pcl/common/centroid.h>
#include <unordered_map>
#include <boost/math/constants/constants.hpp>

namespace descry {

SphericalProjector::SphericalProjector(const Config& cfg, const Perspective& projection) :
    sphere_radius{cfg[config::projection::SPHERE_RAD].as<float>()},
    spiral_divisions{cfg[config::projection::SPIRAL_DIV].as<int>()},
    spiral_turns{cfg[config::projection::SPIRAL_TURNS].as<int>()},
    perspective{projection}
{}

AlignedVector<View> SphericalProjector::generateViews(const FullCloud::ConstPtr& cloud) const {
    using boost::math::float_constants::pi;

    AlignedVector<View> views;
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    for ( int spiral_idx = 0; spiral_idx < spiral_divisions; ++spiral_idx)
    {
        float latitude = -std::acos( std::cos( pi / (2*spiral_turns) )
                                     * (1 - 2 * static_cast<float>(spiral_idx) / spiral_divisions));
        float longitude = spiral_turns * ( pi - 2 * latitude );

        Eigen::Affine3f transformation = Eigen::Affine3f::Identity();

        // Rotate around centroid
        transformation.translate(-centroid.head<3>() + sphere_radius * Eigen::Vector3f::UnitZ());
        transformation.rotate( Eigen::AngleAxisf(latitude, Eigen::Vector3f::UnitX()) );
        transformation.rotate( Eigen::AngleAxisf(longitude, Eigen::Vector3f::UnitZ()) );

        views.emplace_back(project(cloud, transformation.matrix()));
    }

    return views;
}

View SphericalProjector::project(const FullCloud::ConstPtr& full, const Pose& viewpoint) const {
    struct hasher : public std::unary_function<std::pair<int, int>, std::size_t>
    {
        std::size_t operator()(const std::pair<int, int>& k) const
        {
            return (static_cast<size_t>(k.first) << 32) | k.second;
        }
    };

    struct equalizer : public std::binary_function<std::pair<int, int>, std::pair<int, int>, bool>
    {
        bool operator()(const std::pair<int, int>& v0, const std::pair<int, int>& v1) const
        {
            return (v0.first == v1.first) && (v0.second == v1.second);
        }
    };
    std::unordered_map<const std::pair<int, int>, descry::FullPoint, hasher, equalizer> plane_to_space;

    auto min_uv = std::make_pair(std::numeric_limits<int>::max(),
                                 std::numeric_limits<int>::max());
    auto max_uv = std::make_pair(std::numeric_limits<int>::min(),
                                 std::numeric_limits<int>::min());

    for(const auto& point : full->points)
    {
        if (!pcl_isfinite(point.x))
            continue;

        descry::FullPoint transformed = point;
        transformed.getVector4fMap() = viewpoint * point.getVector4fMap();

        if (transformed.z <= 0)
            throw std::runtime_error("Trying to project points with negative distance");

        Eigen::Vector3f projected = perspective * transformed.getVector4fMap();
        auto plane_uv = std::make_pair(
                static_cast<int>(std::round(projected(0) / projected(2))),
                static_cast<int>(std::round(projected(1) / projected(2)))
        );

        if ( plane_to_space.count(plane_uv) == 0 || plane_to_space[plane_uv].z > transformed.z )
            plane_to_space[plane_uv] = transformed;

        min_uv.first = std::min(min_uv.first, plane_uv.first);
        min_uv.second = std::min(min_uv.second, plane_uv.second);
        max_uv.first = std::max(max_uv.first, plane_uv.first);
        max_uv.second = std::max(max_uv.second, plane_uv.second);
    }

    pcl::PointCloud<descry::FullPoint>::Ptr partial(new pcl::PointCloud<descry::FullPoint>());
    partial->width = max_uv.first - min_uv.first + 1;
    partial->height = max_uv.second - min_uv.second + 1;
    partial->points.resize(partial->width * partial->height);
    partial->is_dense = false;

    for (auto v = 0u; v < partial->height; ++v)
        for (auto u = 0u; u < partial->width; ++u)
        {
            descry::FullPoint& point = partial->at(u, v);
            auto uv = std::make_pair(u + min_uv.first, v + min_uv.second);
            if ( plane_to_space.count(uv) )
                point = plane_to_space.at(uv);
            else
                std::fill(std::begin(point.data), std::end(point.data),
                          std::numeric_limits<float>::quiet_NaN());
        }

    //TODO: add radial filter
    return View{ Image{partial}, viewpoint };
}

}
#include <descry/cupcl/conversion.h>
#include <descry/cupcl/unique.h>
#include <thrust/device_vector.h>

namespace {
struct strip_rgba {
    __host__ __device__
    pcl::PointXYZ operator()(const pcl::PointXYZRGBA &other) const {
        return pcl::PointXYZ{other.x, other.y, other.z};
    }
};
}

namespace descry { namespace cupcl {

DualShapeCloud convertToXYZ(const DualConstFullCloud& points) {
    auto d_points_xyz = std::make_unique<DeviceVector2d<pcl::PointXYZ>>(points.device()->getWidth(),
                                                                        points.device()->getHeight());

    thrust::transform(points.device()->getThrust().begin(), points.device()->getThrust().end(),
                      d_points_xyz->getThrust().begin(), strip_rgba{});

    return DualShapeCloud{std::move(d_points_xyz)};
}

}}

#ifndef DESCRY_COMMON_H
#define DESCRY_COMMON_H

#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core.hpp>

#include <descry/cupcl/memory.h>
#include <descry/config.h>
#include <descry/exceptions.h>
#include <pcl/correspondence.h>

namespace descry {

using ShapePoint = pcl::PointXYZ;
using ShapeCloud = pcl::PointCloud<ShapePoint>;
using FullPoint = pcl::PointXYZRGBA;
using FullCloud = pcl::PointCloud<FullPoint>;
using Normals = pcl::PointCloud<pcl::Normal>;
using RefFrames = pcl::PointCloud<pcl::ReferenceFrame>;
using Pose = Eigen::Matrix4f;
using Perspective = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;

template<typename Point, typename... Args>
inline typename pcl::PointCloud<Point>::Ptr make_cloud(Args&&... args) {
    return (typename pcl::PointCloud<Point>::Ptr)(new pcl::PointCloud<Point>(std::forward<Args>(args)...));
}

template<class T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

struct Instances {
    FullCloud::ConstPtr cloud;
    AlignedVector<Pose> poses;
};

// Dual Containers
using DualFullCloud = cupcl::DualContainer<FullPoint>;
using DualConstFullCloud = cupcl::DualContainer<FullPoint, FullCloud::ConstPtr>;

// FIXME: unaligned allocation?
using DualPerpective = cupcl::DualContainer<float, std::unique_ptr<descry::Perspective>>;
using DualShapeCloud = cupcl::DualContainer<pcl::PointXYZ>;
using DualNormals = cupcl::DualContainer<pcl::Normal>;
using DualRefFrames = cupcl::DualContainer<pcl::ReferenceFrame>;

template <typename Descriptor>
struct DescriptorContainerTrait {
    using type = cupcl::DualContainer<Descriptor>;
};

template<>
struct DescriptorContainerTrait<cv::Mat> {
    using type = cv::Mat;
};

template <typename Descriptor>
using DescriptorContainer = typename DescriptorContainerTrait<Descriptor>::type;

template <typename Point>
inline Point make_nan();

template <>
inline pcl::Normal make_nan<pcl::Normal>() {
    auto pt = pcl::Normal{};
    pt.normal_x = pt.normal_y = pt.normal_z = pt.curvature = std::numeric_limits<float>::quiet_NaN();
    return pt;
}

}

#endif //DESCRY_COMMON_H

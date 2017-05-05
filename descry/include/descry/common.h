
#ifndef DESCRY_COMMON_H
#define DESCRY_COMMON_H

#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>

#include <descry/cupcl/memory.h>
#include <descry/exceptions.h>

namespace descry {

using Config = YAML::Node;
using ShapePoint = pcl::PointXYZ;
using ShapeCloud = pcl::PointCloud<ShapePoint>;
using FullPoint = pcl::PointXYZRGBA;
using FullCloud = pcl::PointCloud<FullPoint>;
using Normals = pcl::PointCloud<pcl::Normal>;
using RefFrames = pcl::PointCloud<pcl::ReferenceFrame>;
using Pose = Eigen::Matrix4f;
using Perspective = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;

template<typename Point>
inline typename pcl::PointCloud<Point>::Ptr make_cloud() {
    return (typename pcl::PointCloud<Point>::Ptr)(new pcl::PointCloud<Point>());
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

}

#endif //DESCRY_COMMON_H


#ifndef DESCRY_COMMON_H
#define DESCRY_COMMON_H

#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>

namespace descry {

using Config = YAML::Node;
using Point = pcl::PointXYZRGBA;
using PointCloud = pcl::PointCloud<Point>;
using Normals = pcl::PointCloud<pcl::Normal>;
using RFs = pcl::PointCloud<pcl::ReferenceFrame>;
using Pose = Eigen::Matrix4f;
using Perspective = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;

template<class T>
using AlignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

}

#endif //DESCRY_COMMON_H


#ifndef DESCRY_COMMON_H
#define DESCRY_COMMON_H

#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace descry {

using Point = pcl::PointXYZRGBA;
using PointCloud = pcl::PointCloud<Point>;
using Normals = pcl::PointCloud<pcl::Normal>;
using RFs = pcl::PointCloud<pcl::ReferenceFrame>;
using Pose = Eigen::Matrix4f;
using Perspective = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;

}

#endif //DESCRY_COMMON_H

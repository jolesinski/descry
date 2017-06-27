#ifndef DESCRY_UTILS_H
#define DESCRY_UTILS_H

#include <descry/common.h>

namespace descry {

template<typename Point>
inline long count_finite(const pcl::PointCloud<Point>& cloud) {
    return std::count_if(std::begin(cloud), std::end(cloud), pcl::isFinite<Point>);
}

template<typename Point>
inline float finite_ratio(const pcl::PointCloud<Point>& cloud) {
    return static_cast<float>(count_finite<Point>(cloud)) / cloud.size();
}

template<typename Point>
inline Point make_null_point() {
    auto null_point = Point{};
    null_point.getVector3fMap() = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
    return null_point;
}

template<>
inline FullPoint make_null_point() {
    auto null_point = FullPoint{};
    null_point.getVector3fMap() = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
    null_point.rgba = 0;
    null_point.a = 255;
    return null_point;
}

template<typename Point>
inline void remove_indices(pcl::PointCloud<Point>& inout, const std::vector<int>& indices) {
    auto null_point = make_null_point<FullPoint>();
    inout.is_dense = false;
    for (auto idx : indices)
        inout.at(idx) = null_point;
}

template<typename Point, typename UnaryPredicate>
inline void remove_if(pcl::PointCloud<Point>& inout, UnaryPredicate pred) {
    auto null_point = make_null_point<FullPoint>();
    inout.is_dense = false;
    for (auto& point : inout)
        if (pred(point))
            point = null_point;
}

template<typename Point, typename Point2, typename BinaryPredicate>
inline void remove_if(pcl::PointCloud<Point>& inout, const pcl::PointCloud<Point2>& in2, BinaryPredicate pred) {
    auto null_point = make_null_point<FullPoint>();
    inout.is_dense = false;
    for (auto idx = 0u; idx < inout.size(); ++idx)
        if (pred(inout[idx], in2.at(idx)))
            inout[idx] = null_point;
}

}

#endif //DESCRY_UTILS_H

#ifndef DESCRY_CUPCL_VECTOR_2D_H
#define DESCRY_CUPCL_VECTOR_2D_H

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include <pcl/point_cloud.h>
//#include <descry/common.h>

namespace descry { namespace cupcl {

// To keep descy independent of CUDA compiler, leave incomplete
#ifndef __CUDACC__
template <class Point, class HostContainer>
class DeviceVector2d;
#else

template <class Point, class HostContainer>
class _DeviceVector2dCore {
public:
    Point* getRaw() { return thrust::raw_pointer_cast(&points[0]); }
    const Point* getRaw() const { return thrust::raw_pointer_cast(&points[0]); };

    thrust::device_vector<Point>& getThrust() { return points; };
    const thrust::device_vector<Point>& getThrust() const { return points; };

    std::size_t size() const { return points.size(); }
    std::size_t getWidth() const { return width; }
    std::size_t getHeight() const { return height; }

protected:
    _DeviceVector2dCore(std::size_t width, std::size_t height) : width(width), height(height), points(width*height) {}

    template<class... Args>
    _DeviceVector2dCore(std::size_t width, std::size_t height, Args&&... args) : width(width), height(height), points(args...) {}

    ~_DeviceVector2dCore() = default;

    std::size_t width = 0, height = 0;
    thrust::device_vector<Point> points;
};

template <class Point, class Container = typename pcl::PointCloud<Point>::Ptr>
class DeviceVector2d : public _DeviceVector2dCore<Point, Container> {
public:
    using Ptr = std::unique_ptr<DeviceVector2d<Point, Container>>;

    DeviceVector2d(std::size_t width, std::size_t height)
        : _DeviceVector2dCore<Point, Container>(width, height) {}

    DeviceVector2d(const typename pcl::PointCloud<Point>::ConstPtr& pcl)
        : _DeviceVector2dCore<Point, Container>(pcl->width, pcl->height, pcl->points) {}

    typename pcl::PointCloud<Point>::Ptr download() const {
        auto pcl = typename pcl::PointCloud<Point>::Ptr(new pcl::PointCloud<Point>{});
        pcl->width = this->width;
        pcl->height = this->height;
        pcl->points.resize(this->size());
        thrust::copy(this->points.begin(), this->points.end(), pcl->points.begin());
        return pcl;
    }
};

template <class Scalar, int Rows, int Cols>
using EigenPtr = std::unique_ptr<Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>>;

template <class Point, int Rows, int Cols>
class DeviceVector2d<Point, EigenPtr<Point, Rows, Cols>>
    : public _DeviceVector2dCore<Point, EigenPtr<Point, Rows, Cols>> {
public:
    using Ptr = std::unique_ptr<DeviceVector2d<Point, EigenPtr<Point, Rows, Cols>>>;

    // FIXME: (Rows, Cols)
    DeviceVector2d(std::size_t width, std::size_t height)
        : _DeviceVector2dCore<Point, EigenPtr<Point, Rows, Cols>>(width, height) {}

    DeviceVector2d(const EigenPtr<Point, Rows, Cols>& eigen)
        : _DeviceVector2dCore<Point, EigenPtr<Point, Rows, Cols>>(Cols, Rows, eigen->data(), eigen->data() + eigen->size()) {}

    EigenPtr<Point, Rows, Cols> download() const {
        thrust::host_vector<float> h_vec = this->points;
        return EigenPtr<Point, Rows, Cols>(new Eigen::Matrix<float, Rows, Cols, Eigen::RowMajor>{thrust::raw_pointer_cast(&h_vec[0])});
    }
};

#endif

template<class T>
struct remove_underlying_const_helper {
    using type = T;
};

template<template<class, class...> class T, class I, class... Rest>
struct remove_underlying_const_helper<T<const I, Rest...>> {
    using type = T<I>;
};

template<class T>
using remove_underlying_const = typename remove_underlying_const_helper<T>::type;

}}

#endif //DESCRY_CUPCL_VECTOR_2D_H

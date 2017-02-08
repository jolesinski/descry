#include <descry/common.h>
#include <descry/cupcl/memory.h>
#include <thrust/device_vector.h>
#include <pcl/point_types.h>

// TODO: add support for constptr
// #include <type_traits>

namespace descry { namespace cupcl {

template<class H_T, class H_C, class D_T, class D_C>
DualContainer<H_T, H_C, D_T, D_C>::DualContainer(H_C h) : h_container(std::move(h)) {}

template<class H_T, class H_C, class D_T, class D_C>
DualContainer<H_T, H_C, D_T, D_C>::DualContainer(D_C d) : d_container(std::move(d)) {}

template<class H_T, class H_C, class D_T, class D_C>
DualContainer<H_T, H_C, D_T, D_C>::DualContainer(DualContainer&& other) = default;

template<class H_T, class H_C, class D_T, class D_C>
DualContainer<H_T, H_C, D_T, D_C>& DualContainer<H_T, H_C, D_T, D_C>::operator=(DualContainer<H_T, H_C, D_T, D_C>&& other) = default;

template<class H_T, class H_C, class D_T, class D_C>
DualContainer<H_T, H_C, D_T, D_C>::~DualContainer() = default;

template<class H_T, class H_C, class D_T, class D_C>
void DualContainer<H_T, H_C, D_T, D_C>::clearDevice() {
    d_container.reset();
}

template<class H_T, class H_C, class D_T, class D_C>
void DualContainer<H_T, H_C, D_T, D_C>::clearHost() {
    h_container.reset();
}

template<class H_T, class H_C, class D_T, class D_C>
bool DualContainer<H_T, H_C, D_T, D_C>::isDeviceSet() const {
    return !!d_container;
}

template<class H_T, class H_C, class D_T, class D_C>
bool DualContainer<H_T, H_C, D_T, D_C>::isHostSet() const {
    return !!h_container;
}

template<class H_T, class H_C, class D_T, class D_C>
void DualContainer<H_T, H_C, D_T, D_C>::upload() const {
    if (isHostSet())
        d_container.reset(new thrust::device_vector<D_T>(h_container->points));
}

template<class H_T, class H_C, class D_T, class D_C>
void DualContainer<H_T, H_C, D_T, D_C>::download() const {
    if (isDeviceSet()) {
        h_container.reset(new pcl::PointCloud<H_T>());
        h_container->width = d_container->size();
        h_container->height = 1;
        h_container->points.resize(d_container->size());
        thrust::copy(d_container->begin(), d_container->end(), h_container->points.begin());
    }
}

template<class H_T, class H_C, class D_T, class D_C>
std::size_t DualContainer<H_T, H_C, D_T, D_C>::size() const {
    if (isHostSet())
        return h_container->size();
    if (isDeviceSet())
        return d_container->size();
    return 0;
}

template
class DualContainer<pcl::PointXYZ>;

template
class DualContainer<pcl::PointXYZRGBA>;

template
class DualContainer<pcl::Normal>;

template<>
void DualContainer<float, std::unique_ptr<descry::Perspective>>::upload() const {
    if (isHostSet())
        d_container.reset(new thrust::device_vector<float>(h_container->data(),
                                                           h_container->data() + 12));
}

template<>
void DualContainer<float, std::unique_ptr<descry::Perspective>>::download() const {
    /*if (isDeviceSet()) {
        h_container.reset(new descry::Perspective());

        thrust::copy(d_container->begin(), d_container->end(), h_container->points.begin());
    }*/
}

template
class DualContainer<float, std::unique_ptr<descry::Perspective>>;

}}
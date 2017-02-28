#include <descry/common.h>
#include <descry/cupcl/memory.h>
#include <pcl/point_types.h>

namespace descry { namespace cupcl {

template<class H_T, class H_C, class D_T, class D_C>
DualContainer<H_T, H_C, D_T, D_C>::DualContainer() = default;

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
    assert(isHostSet());
    d_container.reset(new typename D_C::element_type{h_container});
}

template<class H_T, class H_C, class D_T, class D_C>
void DualContainer<H_T, H_C, D_T, D_C>::download() const {
    assert(isDeviceSet());
    h_container = d_container->download();
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
class DualContainer<pcl::PointXYZRGBA>;

template
class DualContainer<pcl::PointXYZ>;

template
class DualContainer<pcl::Normal>;

template
class DualContainer<pcl::ReferenceFrame>;

template
class DualContainer<pcl::FPFHSignature33>;

template
class DualContainer<pcl::SHOT352>;

template
class DualContainer<float, std::unique_ptr<descry::Perspective>>;

template<>
void DualContainer<pcl::PointXYZRGBA, pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr>::download() const {
    // TODO: should throw
}

template
class DualContainer<pcl::PointXYZRGBA, pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr>;


}}
#ifndef DESCRY_CUPCL_MEMORY_H
#define DESCRY_CUPCL_MEMORY_H

#include <descry/cupcl/vector_2d.h>
#include <memory.h>

namespace descry { namespace cupcl {

template<class HostType, class HostContainer = typename pcl::PointCloud<HostType>::Ptr,
         class DeviceType = HostType,
         class DeviceContainer = std::unique_ptr<DeviceVector2d<DeviceType, remove_underlying_const<HostContainer>>>>
class DualContainer {
public:
    DualContainer();
    DualContainer(HostContainer h);
    DualContainer(DeviceContainer d);

    // thrust::device_vector is incomplete, thus destructor needs to be compiled with nvcc
    DualContainer(DualContainer&& other);
    DualContainer& operator=(DualContainer&& other);
    ~DualContainer();

    DualContainer(const DualContainer& other) = delete;
    DualContainer& operator=(const DualContainer& other) = delete;

    void reset() { clearHost(); clearDevice(); }
    void reset(DeviceContainer d) { clearHost(); d_container = std::move(d); }
    void reset(HostContainer h) { clearDevice(); h_container = std::move(h); }

    const DeviceContainer& device() const {
        if (!isDeviceSet())
            upload();
        return d_container;
    }

    const HostContainer& host() const {
        if (!isHostSet())
            download();
        return h_container;
    }

    std::size_t size() const noexcept;
    bool empty() const noexcept { return size() == 0; }

private:
    void upload() const;
    void download() const;

    void clearDevice();
    void clearHost();

    bool isDeviceSet() const noexcept;
    bool isHostSet() const noexcept;

    mutable HostContainer h_container;
    mutable DeviceContainer d_container;
};

#ifdef CUPCL_MOCK
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
void DualContainer<H_T, H_C, D_T, D_C>::clearDevice() {}

template<class H_T, class H_C, class D_T, class D_C>
void DualContainer<H_T, H_C, D_T, D_C>::clearHost() {
    h_container.reset();
}

template<class H_T, class H_C, class D_T, class D_C>
bool DualContainer<H_T, H_C, D_T, D_C>::isDeviceSet() const noexcept { return false; }

template<class H_T, class H_C, class D_T, class D_C>
bool DualContainer<H_T, H_C, D_T, D_C>::isHostSet() const noexcept {
    return !!h_container;
}

template<class H_T, class H_C, class D_T, class D_C>
void DualContainer<H_T, H_C, D_T, D_C>::upload() const {}

template<class H_T, class H_C, class D_T, class D_C>
void DualContainer<H_T, H_C, D_T, D_C>::download() const {}

template<class H_T, class H_C, class D_T, class D_C>
std::size_t DualContainer<H_T, H_C, D_T, D_C>::size() const noexcept {
    if (isHostSet())
        return h_container->size();
    return 0;
}

#endif

} }

#endif //DESCRY_CUPCL_MEMORY_H

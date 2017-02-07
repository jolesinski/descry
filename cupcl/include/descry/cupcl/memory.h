#ifndef DESCRY_CUPCL_MEMORY_H
#define DESCRY_CUPCL_MEMORY_H

#include <memory.h>
#include <pcl/point_cloud.h>

// To keep descy independent of CUDA compiler, thrust is left incomplete
namespace thrust {
template<typename T>
class device_malloc_allocator;
template<typename T, typename Alloc>
class device_vector;
}

namespace descry { namespace cupcl {

template<class HostType, class HostContainer = typename pcl::PointCloud<HostType>::Ptr,
         class DeviceType = HostType, class DeviceContainer = std::unique_ptr<thrust::device_vector<DeviceType, thrust::device_malloc_allocator<DeviceType>>>>
class DualContainer {
public:
    using HostContainerType = HostContainer;
    using DeviceContainerType = DeviceContainer;

    DualContainer(HostContainer h);
    DualContainer(DeviceContainer d);

    // thrust::device_vector is incomplete, thus destructor needs to be compiled with nvcc
    DualContainer(DualContainer&& other);
    DualContainer& operator=(DualContainer&& other);
    ~DualContainer();

    void reset() { clearHost(); clearDevice(); }
    void reset(DeviceContainer d) { clearHost(); d_container = std::move(d); }
    void reset(HostContainer h) { clearDevice(); h_container = std::move(h); }

    void upload() const;
    void download() const;

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

private:
    void clearDevice();
    void clearHost();

    bool isDeviceSet() const;
    bool isHostSet() const;

    mutable HostContainer h_container;
    mutable DeviceContainer d_container;
};

} }

#endif //DESCRY_CUPCL_MEMORY_H

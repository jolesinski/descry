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

    void reset() { clearHost(); clearDevice(); }
    void reset(DeviceContainer d) { clearHost(); d_container = std::move(d); }
    void reset(HostContainer h) { clearDevice(); h_container = std::move(h); }

    // FIXME: those two should be private for const correctness
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

    std::size_t size() const;
    bool empty() const { return size() == 0; }

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

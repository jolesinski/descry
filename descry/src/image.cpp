#include <descry/image.h>

namespace descry {

template<> const PointCloud::ConstPtr& Image::get() const {
    return cloud.host();
}

template<> const Perspective& Image::get() const {
    return *projection.host();
}

}
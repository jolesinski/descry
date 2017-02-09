#ifndef DESCRY_IMAGE_H
#define DESCRY_IMAGE_H

#include <descry/common.h>
#include <descry/cupcl/memory.h>

namespace descry {

class Image {
public:
    Image(PointCloud::ConstPtr cloud) : cloud(cloud) {};
    template <class T> const T& get() const;
private:
    cupcl::DualContainer<const Point, PointCloud::ConstPtr, Point> cloud;
    cupcl::DualContainer<float, std::unique_ptr<descry::Perspective>> projection;
};

}

#endif //DESCRY_IMAGE_H

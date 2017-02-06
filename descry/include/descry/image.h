#ifndef DESCRY_IMAGE_H
#define DESCRY_IMAGE_H

#include <descry/common.h>

namespace descry {

class Image {
public:
    Image(PointCloud::ConstPtr cloud) : cloud(cloud) {};
    template <class T> const T& get() const;
private:
    PointCloud::ConstPtr cloud;
    std::unique_ptr<Perspective> projection;
};

}

#endif //DESCRY_IMAGE_H

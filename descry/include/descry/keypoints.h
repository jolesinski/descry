#ifndef DESCRY_KEYPOINTS_H
#define DESCRY_KEYPOINTS_H

#include <descry/image.h>

namespace descry {

using ShapeKeypoints = descry::ShapeCloud;

class ShapeKeypointDetector {
public:
    bool configure(const Config &config);
    ShapeKeypoints::Ptr compute(const Image &image);

private:
    std::function<ShapeKeypoints::Ptr( const Image& )> _nest;
};

}

#endif //DESCRY_KEYPOINTS_H

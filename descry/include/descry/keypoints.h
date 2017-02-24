#ifndef DESCRY_KEYPOINTS_H
#define DESCRY_KEYPOINTS_H

#include <descry/image.h>
#include <descry/config/keypoints.h>

namespace descry {

using ShapeKeypoints = DualShapeCloud;

class ShapeKeypointDetector {
public:
    bool configure(const Config &config);
    ShapeKeypoints compute(const Image &image);

private:
    std::function<ShapeKeypoints( const Image& )> _nest;
};

}

#endif //DESCRY_KEYPOINTS_H

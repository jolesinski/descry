#ifndef DESCRY_KEYPOINTS_H
#define DESCRY_KEYPOINTS_H

#include <descry/image.h>
#include <descry/config/keypoints.h>

namespace descry {

class KeypointDetector {
public:
    bool configure(const Config &config);
    Image::Keypoints compute(const Image &image) const;

private:
    std::function<Image::Keypoints( const Image& )> nest_;
};

}

#endif //DESCRY_KEYPOINTS_H

#ifndef DESCRY_REF_FRAMES_H
#define DESCRY_REF_FRAMES_H

#include <descry/image.h>
#include <descry/config/ref_frames.h>

namespace descry {

class RefFramesEstimation {
public:
    bool configure(const Config& config);
    DualRefFrames compute(const Image& image) const;
private:
    std::function<DualRefFrames(const Image&)> _est;
};

}

#endif //DESCRY_REF_FRAMES_H

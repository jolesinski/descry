#ifndef DESCRY_REF_FRAMES_H
#define DESCRY_REF_FRAMES_H

#include <descry/image.h>
#include <descry/keypoints.h>
#include <descry/config/ref_frames.h>

namespace descry {

class RefFramesEstimation {
public:
    bool configure(const Config& config);
    bool is_configured() const noexcept;

    DualRefFrames compute(const Image& image, const Keypoints& keys) const;
private:
    std::function<DualRefFrames(const Image&, const Keypoints&)> est_;
};

}

#endif //DESCRY_REF_FRAMES_H

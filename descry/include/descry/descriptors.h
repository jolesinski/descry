#ifndef DESCRY_DESCRIBER_H
#define DESCRY_DESCRIBER_H

#include <descry/config/features.h>
#include <descry/image.h>
#include <descry/keypoints.h>
#include <descry/ref_frames.h>
#include <descry/viewer.h>

namespace descry {

struct KeyFrame {
    using Ptr = std::shared_ptr<KeyFrame>;
    DualRefFrames ref_frames;
    Keypoints keypoints;
};

template <typename Descriptor>
class Description {
public:
    Description() : key_frame(std::make_shared<KeyFrame>()) {}

    const Keypoints& getKeypoints() const {
        return key_frame->keypoints;
    }

    void setKeypoints(const Keypoints& keypoints) {
        key_frame->keypoints = keypoints;
    }

    void setKeypoints(Keypoints&& keypoints) {
        key_frame->keypoints = std::move(keypoints);
    }

    const DualRefFrames& getRefFrames() const {
        return key_frame->ref_frames;
    }

    void setRefFrames(DualRefFrames&& ref_frames) {
        key_frame->ref_frames = std::move(ref_frames);
    }

    const DescriptorContainer<Descriptor>& getFeatures() const {
        return features;
    }

    void setFeatures(DescriptorContainer<Descriptor>&& feats) {
        this->features = std::move(feats);
    }

    std::shared_ptr<KeyFrame> getKeyFrame() {
        return key_frame;
    }

protected:
    DescriptorContainer<Descriptor> features;
    std::shared_ptr<KeyFrame> key_frame;
};

template <typename Descriptor>
class Describer {
public:
    bool configure(const Config &cfg);
    Description<Descriptor> compute(const Image& image);

private:
    std::function<Description<Descriptor>( const Image& )> _descr;
    Viewer<Keypoints> viewer_;
    bool log_latency_ = false;
};

}

#endif //DESCRY_DESCRIBER_H

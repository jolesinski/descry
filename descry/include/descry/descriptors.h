#ifndef DESCRY_DESCRIBER_H
#define DESCRY_DESCRIBER_H

#include <descry/config/features.h>
#include <descry/image.h>
#include <descry/keypoints.h>
#include <descry/ref_frames.h>
#include <descry/viewer.h>

namespace descry {

struct KeyFrameHandle {
    Keypoints* keys;
    DualRefFrames* rfs;
};

template <typename Descriptor>
class Description {
public:
    const Keypoints& getKeypoints() const {
        return keypoints;
    }

    void setKeypoints(const Keypoints& keypoints) {
        this->keypoints = keypoints;
    }

    void setKeypoints(Keypoints&& keypoints) {
        this->keypoints = std::move(keypoints);
    }

    const DualRefFrames& getRefFrames() const {
        return ref_frames;
    }

    void setRefFrames(DualRefFrames&& ref_frames) {
        this->ref_frames = std::move(ref_frames);
    }

    const DescriptorContainer<Descriptor>& getFeatures() const {
        return features;
    }

    void setFeatures(DescriptorContainer<Descriptor>&& feats) {
        this->features = std::move(feats);
    }

    KeyFrameHandle getKeyFrame() {
        return {&keypoints, &ref_frames};
    }

protected:
    DualRefFrames ref_frames;
    Keypoints keypoints;
    DescriptorContainer<Descriptor> features;
};

template <typename Descriptor>
class Describer {
public:
    bool configure(const Config &config);
    Description<Descriptor> compute(const Image& image);

private:
    std::function<Description<Descriptor>( const Image& )> _descr;
    Viewer<Keypoints> viewer_;
};

}

#endif //DESCRY_DESCRIBER_H

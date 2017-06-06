#ifndef DESCRY_VIEWER_H
#define DESCRY_VIEWER_H

#include <descry/normals.h>
#include <descry/alignment.h>
#include <descry/matching.h>
#include <descry/clusters.h>
#include <descry/config/viewer.h>

namespace descry {

template <typename Step>
class Viewer {};

template <>
class Viewer<NormalEstimation> {
public:
    void show(const Image& image);
};

template <>
class Viewer<KeypointDetector> {
public:
    void show(const Image& image, const Keypoints& keypoints);
};

template <>
class Viewer<Aligner> {
public:
    void configure(const Config& cfg);
    void show(const FullCloud::ConstPtr& scene, const Instances& instances);
private:
    Config cfg_;
};

}

#endif //DESCRY_VIEWER_H

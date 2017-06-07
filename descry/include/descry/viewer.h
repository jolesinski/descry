#ifndef DESCRY_VIEWER_H
#define DESCRY_VIEWER_H

#include <descry/alignment.h>
#include <descry/matching.h>
#include <descry/clusters.h>
#include <descry/config/viewer.h>

namespace descry {


template <typename Step>
class Viewer {
};

template <typename Step>
class ConfigurableViewer {
public:
    void configure(const Config& cfg) {
        if (cfg[config::viewer::NODE_NAME])
            cfg_ = cfg[config::viewer::NODE_NAME];
    }
protected:
    Config cfg_;
};

template <>
class Viewer<Normals> : public ConfigurableViewer<Normals> {
public:
    void show(const FullCloud::ConstPtr& image, const Normals::ConstPtr& normals) const;
};

template <>
class Viewer<KeypointDetector> : public ConfigurableViewer<KeypointDetector> {
public:
    void show(const Image& image, const Keypoints& keypoints) const;
};

template <>
class Viewer<Aligner> : public ConfigurableViewer<Aligner> {
public:
    void show(const FullCloud::ConstPtr& scene, const Instances& instances) const;
};

}

#endif //DESCRY_VIEWER_H

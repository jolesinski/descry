#ifndef DESCRY_VIEWER_H
#define DESCRY_VIEWER_H

#include <descry/common.h>
#include <descry/config/viewer.h>
#include <descry/image.h>

namespace descry {

template <typename Step>
class Viewer {
};

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
class Viewer<Normals> : public ConfigurableViewer {
public:
    void show(const FullCloud::ConstPtr& image, const Normals::ConstPtr& normals) const;
};

class Keypoints;
template <>
class Viewer<Keypoints> : public ConfigurableViewer {
public:
    void show(const Image& image, const Keypoints& keypoints);
protected:
    unsigned int show_count_ = 0u;
};

class Clusterizer;
class KeyFrameHandle;
class Model;
class ViewerStorage;
template <>
class Viewer<Clusterizer> : public ConfigurableViewer {
public:
    void addModel(const Model& model, const std::vector<KeyFrameHandle>& keyframes);
    void show(const Image& scene, const KeyFrameHandle& keyframe,
              const std::vector<pcl::CorrespondencesPtr>& corrs);
protected:
    std::function<void(const Image&, const KeyFrameHandle&,
                       const std::vector<pcl::CorrespondencesPtr>&)> show_;
};

class Aligner;
template <>
class Viewer<Aligner> : public ConfigurableViewer {
public:
    void show(const FullCloud::ConstPtr& scene, const Instances& instances) const;
};

}

#endif //DESCRY_VIEWER_H

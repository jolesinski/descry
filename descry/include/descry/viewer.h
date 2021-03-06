#ifndef DESCRY_VIEWER_H
#define DESCRY_VIEWER_H

#include <descry/common.h>
#include <descry/config/viewer.h>
#include <descry/image.h>
#include <pcl/pcl_base.h>

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
class Viewer<Image> : public ConfigurableViewer {
public:
    void show(const Image& image) const;
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
class KeyFrame;
class Model;
class ModelSceneMatches;
template <>
class Viewer<Clusterizer> : public ConfigurableViewer {
public:
    void addModel(const Model& model, const std::vector<std::shared_ptr<KeyFrame>>& keyframes);
    void show(const Image& scene, const ModelSceneMatches& matches);
    void show(const Image& scene, const KeyFrame& keyframe,
              const std::vector<pcl::Correspondences>& clustered, unsigned int idx);
protected:
    std::function<void(const Image&, const ModelSceneMatches&)> show_matches_;
    std::function<void(const Image&, const KeyFrame&, const std::vector<pcl::Correspondences>&, unsigned int idx)> show_clusters_;
};

class Segmenter;
template <>
class Viewer<Segmenter> : public ConfigurableViewer {
public:
    void show(const Image& image, const std::vector<pcl::IndicesPtr>& segments) const;
};

class Aligner;
template <>
class Viewer<Aligner> : public ConfigurableViewer {
public:
    void show(const FullCloud::ConstPtr& scene, const Instances& instances) const;
};

}

#endif //DESCRY_VIEWER_H

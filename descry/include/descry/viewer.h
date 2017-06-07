#ifndef DESCRY_VIEWER_H
#define DESCRY_VIEWER_H

#include <descry/common.h>
#include <descry/config/viewer.h>
#include <descry/image.h>

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

class Keypoints;
template <>
class Viewer<Keypoints> : public ConfigurableViewer<Keypoints> {
public:
    void show(const Image& image, const Keypoints& keypoints);
private:
    unsigned int show_count_ = 0u;
};

class Aligner;
template <>
class Viewer<Aligner> : public ConfigurableViewer<Aligner> {
public:
    void show(const FullCloud::ConstPtr& scene, const Instances& instances) const;
};

}

#endif //DESCRY_VIEWER_H

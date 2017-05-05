#include <descry/preprocess.h>

#include <descry/keypoints.h>
#include <descry/normals.h>
#include <descry/ref_frames.h>

namespace descry {

bool Preprocess::configure(const Config& cfg) {
    auto& normals_cfg = cfg[config::normals::NODE_NAME];
    if (normals_cfg) {
        auto nest = descry::NormalEstimation{};

        if(!nest.configure(normals_cfg))
            return false;

        steps_.emplace_back( [nest{std::move(nest)}]
                (Image& image) { image.setNormals(nest.compute(image)); }
        );
    }

    auto& keys_cfg = cfg[config::keypoints::NODE_NAME];
    if (keys_cfg) {
        auto kdet = descry::KeypointDetector{};

        if(!kdet.configure(keys_cfg))
            return false;

        steps_.emplace_back( [kdet{std::move(kdet)}]
                (Image& image) { image.setKeypoints(kdet.compute(image)); }
        );
    }

    auto& rfs_cfg = cfg[config::ref_frames::NODE_NAME];
    if (rfs_cfg) {
        auto rfest = descry::RefFramesEstimation{};

        if(!rfest.configure(rfs_cfg))
            return false;

        steps_.emplace_back( [rfest{std::move(rfest)}]
                (Image& image) { image.setRefFrames(rfest.compute(image)); }
        );
    }

    return true;
}

void Preprocess::process(Image& image) const {
    for(auto& step : steps_)
        step(image);
}

}

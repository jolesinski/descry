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

    return true;
}

void Preprocess::process(Image& image) const {
    for(auto& step : steps_)
        step(image);
}

}

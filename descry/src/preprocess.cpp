#include <descry/preprocess.h>

#include <descry/keypoints.h>
#include <descry/normals.h>
#include <descry/ref_frames.h>

#include <pcl/filters/passthrough.h>

namespace descry {

bool Preprocess::configure(const Config& cfg) {
    auto& passthrough_cfg = cfg[config::preprocess::PASSTHROUGH];
    if (passthrough_cfg) {
        pcl::PassThrough<descry::FullPoint> pass;
        pass.setKeepOrganized(true);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.0, passthrough_cfg.as<float>());
        filtering_.emplace_back( [pass] (const Image& image) mutable {
            auto filtered = make_cloud<descry::FullPoint>();
            pass.setInputCloud(image.getFullCloud().host());
            pass.filter(*filtered);
            return Image{filtered};
        });
    }

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

Image Preprocess::filter(const descry::FullCloud::ConstPtr& scene) const {
    Image filtered_image{scene};
    for(auto& filter : filtering_)
        filtered_image = filter(filtered_image);

    return filtered_image;
}

void Preprocess::process(Image& image) const {
    for(auto& step : steps_)
        step(image);
}

}

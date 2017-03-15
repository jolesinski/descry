#include <descry/ref_frames.h>

#include <pcl/features/board.h>

using EstBOARD = pcl::BOARDLocalReferenceFrameEstimation<descry::ShapePoint, pcl::Normal, pcl::ReferenceFrame>;

namespace YAML {
template<>
struct convert<EstBOARD> {
    static bool decode(const Node &node, EstBOARD &rhs) {
        if (!node.IsMap()) {
            return false;
        }

        if (node[descry::config::ref_frames::SUPPORT_RAD])
            rhs.setRadiusSearch(node[descry::config::ref_frames::SUPPORT_RAD].as<double>());
        else
            return false;

        if (node[descry::config::ref_frames::BOARD_FIND_HOLES])
            rhs.setFindHoles(node[descry::config::ref_frames::BOARD_FIND_HOLES].as<bool>());

        return true;
    }
};
}

namespace descry {

bool RefFramesEstimation::configure(const Config& config) {
    if (!config["type"])
        return false;

    auto est_type = config["type"].as<std::string>();

    try {
        if (est_type == config::ref_frames::BOARD_TYPE) {
            auto est = config.as<EstBOARD>();
            _est = [ est{std::move(est)} ]
                   (const Image &image) mutable {
                       est.setInputCloud(image.getShapeKeypoints().host());
                       est.setInputNormals(image.getNormals().host());
                       est.setSearchSurface(image.getShapeCloud().host());
                       RefFrames::Ptr ref_frames{new RefFrames{}};
                       est.compute(*ref_frames);
                       return DualRefFrames{ref_frames};
                   };
        } else
            return false;
    } catch ( const YAML::BadConversion& e) {
        return false;
    }

    return true;
}

DualRefFrames RefFramesEstimation::compute(const Image& image) const {
    assert(_est);
    return _est(image);
}

}

#include <descry/model.h>

namespace {

}

namespace descry {

Model::Model(const FullCloud::ConstPtr& full, const Projector& projector) :
        full_(full), views_(projector.generateViews(full)) {
}

Model::Model(const FullCloud::ConstPtr& full, AlignedVector<View>&& views) :
        full_(full), views_(std::move(views)) {
}

void Model::prepare(const Preprocess& preprocessor) {
    for(auto& view : views_)
        preprocessor.process(view.image);
}

namespace {

ShapePoint transformViewKeypoint(const ShapePoint key, const View& view) {
    ShapePoint pt;
    pt.getVector4fMap() = view.viewpoint * key.getVector4fMap();
    return pt;
}

void mergeKeypoints(const View& view, ShapeCloud& merged) {
    auto keys = view.image.getKeypoints().getShape().host();

    merged.points.reserve(merged.size() + keys->size());
    merged.width += keys->size();
    merged.height = 1;
    for (auto key : *keys)
        merged.points.emplace_back(transformViewKeypoint(key, view));
}

}

ShapeCloud::Ptr Model::getFullKeypoints() const {
    auto merged_keys = descry::make_cloud<descry::ShapePoint>();

    for (auto& view : views_)
        mergeKeypoints(view, *merged_keys);

    return merged_keys;
}

//const RefFrames::ConstPtr& Model::getFullRefFrames() const {}

}
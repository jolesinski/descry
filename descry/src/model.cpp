#include <descry/model.h>

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

}
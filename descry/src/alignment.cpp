#include <descry/alignment.h>

namespace descry {

void Aligner::configure(const Config& cfg) {
    if (!cfg[config::TYPE_NODE])
        DESCRY_THROW(InvalidConfigException, "missing aligner type");

    try {
        auto est_type = cfg[config::TYPE_NODE].as<std::string>();
        if (est_type == config::aligner::SPARSE_TYPE) {
            viewer_.configure(cfg);
            matching_.configure(cfg[config::matcher::NODE_NAME]);
            clustering_.configure(cfg[config::clusters::NODE_NAME]);
        } else
            DESCRY_THROW(InvalidConfigException, "unsupported aligner type");
    } catch ( const YAML::RepresentationException& e) {
        DESCRY_THROW(InvalidConfigException, e.what());
    }
}

void Aligner::train(const Model& model) {
    auto key_frames = matching_.train(model);
    clustering_.train(model, key_frames);
}

Instances Aligner::compute(const Image& image) {
    auto model_scene_matches = matching_.match(image);
    auto instances = clustering_.compute(image, model_scene_matches);
    viewer_.show(image.getFullCloud().host(), instances);
    return instances;
}

}

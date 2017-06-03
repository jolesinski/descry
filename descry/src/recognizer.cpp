#include <descry/recognizer.h>

bool descry::Recognizer::configure(const descry::Config &cfg) {
    if (!cfg.IsDefined())
        DESCRY_THROW(InvalidConfigException, "undefined node");

    if (cfg[config::normals::NODE_NAME])
        nest_.configure(cfg[config::normals::NODE_NAME]);

    if (cfg[config::aligner::NODE_NAME])
        aligner_.configure(cfg[config::aligner::NODE_NAME]);
    else
        DESCRY_THROW(InvalidConfigException, "missing aligner config");

    {
        auto refiner_cfg = cfg[config::refiner::NODE_NAME];
        if (refiner_cfg)
            refiner_.configure(refiner_cfg);
    }

    {
        auto verifier_cfg = cfg[config::verifier::NODE_NAME];
        if (verifier_cfg)
            verifier_.configure(verifier_cfg);
    }

    return true;
}

void descry::Recognizer::train(const descry::Model& model) {
    aligner_.train(model);

    if (refiner_.is_configured())
        refiner_.train(model);
}

descry::Instances
descry::Recognizer::compute(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &scene) {
    auto image = descry::Image{scene};

    if (nest_.is_configured())
        image.setNormals(nest_.compute(image));

    auto instances = aligner_.compute(image);

    if (refiner_.is_trained())
        instances = refiner_.compute(image, instances);

    if (verifier_.is_configured())
        instances = verifier_.compute(image, instances);

    return instances;
}


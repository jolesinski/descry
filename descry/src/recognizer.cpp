#include <descry/recognizer.h>
#include <descry/latency.h>

bool descry::Recognizer::configure(const descry::Config &cfg) {
    if (!cfg.IsDefined())
        DESCRY_THROW(InvalidConfigException, "undefined node");

    {
        auto preproc_cfg = cfg[config::preprocess::NODE_NAME];
        if (preproc_cfg) {
            model_preproc_.configure(preproc_cfg[config::MODEL_NODE]);
            scene_preproc_.configure(preproc_cfg[config::SCENE_NODE]);
        }
    }

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

    log_latency_ = cfg[config::LOG_LATENCY].as<bool>(false);

    return true;
}

void descry::Recognizer::train(descry::Model& model) {
    model.prepare(model_preproc_);

    aligner_.train(model);

    if (refiner_.is_configured())
        refiner_.train(model);
}

descry::Instances
descry::Recognizer::compute(const descry::FullCloud::ConstPtr& scene) {
    auto image = Image{scene};

    auto latency = measure_latency(config::preprocess::NODE_NAME, log_latency_);
    scene_preproc_.process(image);

    latency.restart(config::aligner::NODE_NAME);
    auto instances = aligner_.compute(image);
    latency.finish();

    if (refiner_.is_trained()) {
        latency.start(config::refiner::NODE_NAME);
        instances = refiner_.compute(image, instances);
        latency.finish();
    }

    if (verifier_.is_configured()) {
        latency.start(config::verifier::NODE_NAME);
        instances = verifier_.compute(image, instances);
        latency.finish();
    }

    return instances;
}


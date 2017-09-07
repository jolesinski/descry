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

    {
        auto aligner_cfg = cfg[config::aligner::NODE_NAME];
        aligners_.clear();
        if (aligner_cfg.IsMap()) {
            aligners_.emplace_back();
            aligners_.back().configure(aligner_cfg);
        } else if (aligner_cfg.IsSequence()) {
            for (auto it = aligner_cfg.begin(); it != aligner_cfg.end(); ++it) {
                aligners_.emplace_back();
                aligners_.back().configure(*it);
            }
        } else
            DESCRY_THROW(InvalidConfigException, "missing aligner config");
    }

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

    for (auto& aligner : aligners_)
        aligner.train(model);

    if (refiner_.is_configured())
        refiner_.train(model);
}

descry::Instances
descry::Recognizer::compute(const descry::FullCloud::ConstPtr& scene) {
    auto image = Image{scene};

    auto latency = measure_latency(config::preprocess::NODE_NAME, log_latency_);
    scene_preproc_.process(image);

    auto instances = Instances{};
    for (auto& aligner : aligners_) {
        latency.restart(config::aligner::NODE_NAME);
        auto aligned_instances = aligner.compute(image);
        instances.cloud = aligned_instances.cloud;
        instances.poses.insert(instances.poses.end(), aligned_instances.poses.begin(), aligned_instances.poses.end());
        latency.finish();
    }

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


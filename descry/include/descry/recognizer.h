
#ifndef DESCRY_RECOGNIZER_H
#define DESCRY_RECOGNIZER_H

#include <descry/common.h>
#include <descry/model.h>

#include <descry/alignment.h>
#include <descry/normals.h>
#include <descry/refinement.h>
#include <descry/verification.h>

namespace descry {

class Recognizer {
public:
    bool configure(const Config& config);
    void train(Model& model);
    Instances compute(const FullCloud::ConstPtr &scene);
private:
    Preprocess model_preproc_;
    Preprocess scene_preproc_;
    std::vector<Aligner> aligners_;
    Refiner refiner_;
    Verifier verifier_;
    bool log_latency_;
};

}

#endif //DESCRY_RECOGNIZER_H

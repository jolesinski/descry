#ifndef DESCRY_ALIGNMENT_H
#define DESCRY_ALIGNMENT_H

#include <descry/clusters.h>
#include <descry/config/aligner.h>
#include <descry/image.h>
#include <descry/matching.h>
#include <descry/model.h>
#include <descry/viewer.h>

namespace descry {

class Aligner {
public:
    void configure(const Config& config);
    void train(const Model& model);

    Instances compute(const Image& image);
protected:
    std::vector<KeyFrame::Ptr> folded_kfs_;
    std::vector<Matching> matchings_;
    Clusterizer clustering_;
    Viewer<Aligner> viewer_;
};

}

#endif //DESCRY_ALIGNMENT_H

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
    Aligner() = default;
    Aligner(Aligner&& other) = default;
    Aligner& operator=(Aligner&& other) = default;

    void configure(const Config& config);
    virtual void train(const Model& model);
    virtual Instances compute(const Image& image);
private:
    std::unique_ptr<Aligner> strategy_;
};

}

#endif //DESCRY_ALIGNMENT_H

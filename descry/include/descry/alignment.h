#ifndef DESCRY_ALIGNMENT_H
#define DESCRY_ALIGNMENT_H

#include <descry/image.h>
#include <descry/model.h>
#include <descry/config/aligner.h>

namespace descry {

class Aligner {
public:
    class AlignmentStrategy {
    public:
        virtual void setModel(const Model& model) = 0;
        virtual Instances match(const Image& image) = 0;
        virtual ~AlignmentStrategy() = default;
    };

    void configure(const Config& config);
    void setModel(const Model& model);

    Instances compute(const Image& image);
private:
    std::unique_ptr<AlignmentStrategy> strategy_;
};

}

#endif //DESCRY_ALIGNMENT_H

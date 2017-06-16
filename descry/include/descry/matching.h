#ifndef DESCRY_MATCHING_H
#define DESCRY_MATCHING_H

#include <descry/image.h>
#include <descry/model.h>
#include <descry/descriptors.h>
#include <descry/config/matcher.h>
#include <pcl/correspondence.h>

// TODO:
// add thrust reduce min distance strategy
namespace descry {

struct KeyFrameMatches {
    pcl::CorrespondencesPtr corrs;
    KeyFrame::Ptr model;
    KeyFrame::Ptr scene;
};

class Matching {
public:
    class Strategy {
    public:
        virtual void train(const Model& model) = 0;
        virtual std::vector<KeyFrameMatches> match(const Image& image) = 0;
        virtual ~Strategy() = default;
    };

    void configure(const Config& config);
    void train(const Model& model);

    std::vector<KeyFrameMatches> match(const Image& image);
private:
    std::unique_ptr<Strategy> strategy_ = nullptr;
};

}

#endif //DESCRY_MATCHING_H

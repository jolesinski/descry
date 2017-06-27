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

struct ModelSceneMatches {
    struct ViewCorrespondences {
        pcl::CorrespondencesPtr corrs;
        KeyFrame::Ptr view;
    };
    std::vector<ViewCorrespondences> view_corrs;
    KeyFrame::Ptr scene;
};

class Matching {
public:
    Matching() = default;
    Matching(Matching&& other) = default;
    Matching& operator=(Matching&& other) = default;

    void configure(const Config& config);
    virtual std::vector<KeyFrame::Ptr> train(const Model& model);
    virtual ModelSceneMatches match(const Image& image);

    virtual ~Matching() = default;
private:
    std::unique_ptr<Matching> strategy_ = nullptr;
};

}

#endif //DESCRY_MATCHING_H

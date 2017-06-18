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
    class Strategy {
    public:
        virtual std::vector<KeyFrame::Ptr> train(const Model& model) = 0;
        virtual ModelSceneMatches match(const Image& image) = 0;
        virtual ~Strategy() = default;
    };

    void configure(const Config& config);
    std::vector<KeyFrame::Ptr> train(const Model& model);

    ModelSceneMatches match(const Image& image);
private:
    std::unique_ptr<Strategy> strategy_ = nullptr;
};

}

#endif //DESCRY_MATCHING_H

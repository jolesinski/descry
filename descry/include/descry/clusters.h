#ifndef DESCRY_CLUSTERS_H
#define DESCRY_CLUSTERS_H

#include <descry/image.h>
#include <descry/model.h>
#include <descry/matching.h>
#include <descry/config/clusters.h>
#include <pcl/correspondence.h>

namespace descry {

class Clusterizer {
public:
    Clusterizer() = default;
    Clusterizer(Clusterizer&& other) = default;
    Clusterizer& operator=(Clusterizer&& other) = default;

    bool configure(const Config &config);

    virtual void train(const Model& image, const std::vector<KeyFrame::Ptr>& view_keyframes);
    virtual Instances compute(const Image& image, const ModelSceneMatches& matches);

    virtual ~Clusterizer() = default;
private:
    std::unique_ptr<Clusterizer> strategy_ = nullptr;
};

}

#endif //DESCRY_CLUSTERS_H

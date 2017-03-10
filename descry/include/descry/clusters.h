#ifndef DESCRY_CLUSTERS_H
#define DESCRY_CLUSTERS_H

#include <descry/image.h>
#include <descry/model.h>
#include <descry/config/clusters.h>
#include <pcl/correspondence.h>

namespace descry {

class Clusterizer {
public:
    class Strategy {
    public:
        virtual void setModel(const Model& model) = 0;
        virtual Instances compute(const Image& image, const std::vector<pcl::CorrespondencesPtr>& corrs) = 0;
        virtual ~Strategy() = default;
    };

    bool configure(const Config &config);
    void setModel(const Model& image);

    Instances compute(const Image& image, const std::vector<pcl::CorrespondencesPtr>& corrs);

private:
    std::unique_ptr<Strategy> strategy_ = nullptr;
};

}

#endif //DESCRY_CLUSTERS_H

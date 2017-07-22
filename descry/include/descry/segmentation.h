#ifndef DESCRY_SEGMENTATION_H
#define DESCRY_SEGMENTATION_H

#include <descry/common.h>
#include <descry/model.h>
#include <descry/config/segments.h>
#include <pcl/pcl_base.h>

namespace descry {

using Segment = pcl::IndicesPtr;

class Segmenter {
public:
    void configure(const Config& cfg);
    void train(const Model& model);
    std::vector<Segment> compute(const Image& image);
private:
    float distance_threshold_ = 0.02;
    float angular_threshold_ = 0.05;
    unsigned int min_cluster_size_ = 1000;
    unsigned int max_cluster_size_ = 25000;
    Viewer<Segmenter> viewer_;
};

}

#endif //DESCRY_SEGMENTATION_H

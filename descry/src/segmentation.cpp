#include <descry/segmentation.h>

#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/segmentation/euclidean_plane_coefficient_comparator.h>
#include <boost/make_shared.hpp>

#include <descry/viewer.h>
// FIXME: #include <pcl/segmentation/extract_clusters.h>

namespace descry {

void Segmenter::configure(const Config& cfg) {
    distance_threshold_ = cfg[config::segments::DISTNACE_THRESHOLD].as<float>(distance_threshold_);
    angular_threshold_ = cfg[config::segments::ANGULAR_THRESHOLD].as<float>(angular_threshold_);
    min_cluster_size_ = cfg[config::segments::MIN_CLUSTER_SIZE].as<unsigned int>(min_cluster_size_);
    max_cluster_size_ = cfg[config::segments::MAX_CLUSTER_SIZE].as<unsigned int>(max_cluster_size_);

    viewer_.configure(cfg);
}

void Segmenter::train(const Model&) {

}

std::vector<Segment> Segmenter::compute(const Image& image) {
    auto comparator = boost::make_shared<pcl::EuclideanPlaneCoefficientComparator<FullPoint,pcl::Normal>>();
    comparator->setInputCloud(image.getFullCloud().host());
    comparator->setInputNormals(image.getNormals().host());
    comparator->setDistanceThreshold(distance_threshold_);
    comparator->setAngularThreshold(angular_threshold_);

    auto extractor = pcl::OrganizedConnectedComponentSegmentation<FullPoint, pcl::Label>(comparator);
    extractor.setInputCloud(image.getFullCloud().host());

    auto labels = pcl::PointCloud<pcl::Label>{};
    auto indices = std::vector<pcl::PointIndices>{};
    extractor.segment(labels, indices);

    auto segments = std::vector<Segment>{};
    for (auto& segment : indices)
        if (segment.indices.size() > min_cluster_size_ && segment.indices.size() < max_cluster_size_)
            segments.emplace_back(segment.indices);

    viewer_.show(image, segments);
    return segments;
}

}
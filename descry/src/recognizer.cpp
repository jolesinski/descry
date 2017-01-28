#include <descry/recognizer.h>

void descry::Recognizer::configure(const descry::Config &config) {

}

void descry::Recognizer::train(const descry::Model &model) {

}

std::vector<descry::ModelInstance>
descry::Recognizer::recognize(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &scene) {
    return std::vector<descry::ModelInstance>();
}

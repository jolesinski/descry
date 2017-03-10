#include <descry/recognizer.h>
#include <string>

bool descry::Recognizer::configure(const descry::Config &config) {
    if(!config["recognizer"])
        return false;

    auto rec_node = config["recognizer"];
    if(!rec_node["type"])
        return false;

    auto rec_type = rec_node["type"].as<std::string>();
    if(!config[rec_type])
        return false;

    return true;
}

void descry::Recognizer::train(const descry::Model &model) {

}

descry::Instances
descry::Recognizer::recognize(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &scene) {
    return descry::Instances{};
}

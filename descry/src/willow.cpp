#include <descry/willow.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/range/iterator_range.hpp>

#include <pcl/io/pcd_io.h>

#include <algorithm>

namespace descry {

namespace {

template <class T>
std::vector<T> loadNumericData(const std::string& path) {
    std::vector<T> elems;
    std::ifstream infile(path);

    if ( infile.bad() )
        throw std::runtime_error("Unable to load data from " + path);

    std::istream_iterator<T> eos;
    std::istream_iterator<T> iit(infile);

    std::copy(iit, eos, std::back_inserter(elems));

    return elems;
}

FullCloud::Ptr loadCloud(const std::string& cloud_path) {
    FullCloud::Ptr cloud_ptr(new FullCloud);
    if( pcl::io::loadPCDFile<pcl::PointXYZRGBA>(cloud_path, *cloud_ptr) == -1 )
        throw std::runtime_error("Unable to load pcd from" + cloud_path);

    cloud_ptr->sensor_orientation_ = Eigen::Quaternionf::Identity();
    cloud_ptr->sensor_origin_ = Eigen::Vector4f::Zero(4);

    return cloud_ptr;
}

Pose loadPose(const std::string& pose_path) {
    auto elems = loadNumericData<float>(pose_path);
    if ( elems.size() != 16 )
        throw std::runtime_error("No pose found in " + pose_path);

    Eigen::Matrix4f pose(elems.data());
    return pose.transpose();
}

std::vector<int> loadIndices(const std::string& indices_path) {
    auto indices = loadNumericData<int>(indices_path);
    if ( indices.empty() )
        throw std::runtime_error("No indices found in " + indices_path);

    return indices;
}

}

WillowProjector::WillowProjector(const Config& model_cfg) : views_cfg_{model_cfg} {}

AlignedVector<View> WillowProjector::generateViews(const FullCloud::ConstPtr& /*cloud*/) const {
    auto views = AlignedVector<View>{};
    views.reserve(views_cfg_.size());
    for (const auto& view_cfg : views_cfg_) {
        const auto& cloud_path = view_cfg["cloud"].as<std::string>();
        const auto& indices_path = view_cfg["indices"].as<std::string>();
        const auto& pose_path = view_cfg["pose"].as<std::string>();

        views.emplace_back(loadView(cloud_path, indices_path, pose_path));
    }

    return views;
}

View WillowProjector::loadView(const std::string& cloud_path,
                               const std::string& indices_path,
                               const std::string& pose_path) const {
    auto cloud = loadCloud(cloud_path);
    auto indices = std::vector<int>(loadIndices(indices_path));
    auto viewpoint = loadPose(pose_path);

    auto null_point = FullPoint{};
    null_point.getVector3fMap() = Eigen::Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
    auto filtered_cloud = make_cloud<FullPoint>(cloud->width, cloud->height, null_point );
    for (auto idx : indices) {
        filtered_cloud->at(idx) = cloud->at(idx);
    }

    return View{ Image{filtered_cloud}, indices, viewpoint };
}


WillowDatabase::WillowDatabase(const Config& db_cfg) : db_cfg_{db_cfg} {}

Model WillowDatabase::loadModel(const std::string& model_name) const {
    const auto& model_cfg = db_cfg_[model_name];

    auto projector = descry::WillowProjector(model_cfg["views"]);
    auto full = loadCloud(model_cfg["full"].as<std::string>());

    return Model{full, projector};
}

std::unordered_map<std::string, Model> WillowDatabase::loadDatabase() const {
    auto models = std::unordered_map<std::string, Model>{};
    for (const auto& model_cfg : db_cfg_) {
        const auto model_name = model_cfg.first.as<std::string>();
        models.emplace(model_name, loadModel(model_name));
    }

    return models;
}


std::vector<WillowTestSet::AnnotatedScene>
WillowTestSet::loadTest(const Config& test_cfg, std::size_t max_frames) const {
    auto annotated_scenes = std::vector<AnnotatedScene>{};
    annotated_scenes.reserve(test_cfg.size());
    for (const auto& scene : test_cfg) {
        if (max_frames-- == 0)
            break;
        annotated_scenes.emplace_back(loadCloud(scene["cloud"].as<std::string>()),
                                      loadInstances(scene["instances"]));
    }

    return annotated_scenes;
}

WillowTestSet::InstanceMap
WillowTestSet::loadInstances(const Config& instances_cfg) const {
    auto instances = InstanceMap{};
    for (const auto& instance_cfg : instances_cfg) {
        const auto& object_name = instance_cfg["object"].as<std::string>();
        const auto& pose_path = instance_cfg["pose"].as<std::string>();

        auto pose = loadPose(pose_path);

        auto instance = instances.find(object_name);
        if(instance == instances.end())
            instances.insert({object_name, {pose}});
        else
            instance->second.emplace_back(pose);
    }

    return instances;
}

}
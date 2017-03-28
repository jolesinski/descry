#include <descry/willow.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/range/iterator_range.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>

#include <algorithm>

namespace descry {

namespace {

template<class PredT>
std::vector<boost::filesystem::path> getFiles(const std::string& dir_path, PredT pred) {
    namespace fs = boost::filesystem;

    auto filenames = std::vector<boost::filesystem::path>{};
    for (auto dirent : boost::make_iterator_range(fs::directory_iterator(dir_path),
                                                  fs::directory_iterator())) {
        if( pred(dirent.path()) )
            filenames.emplace_back(dirent.path());
    }

    return filenames;
}

std::vector<boost::filesystem::path> getCloudFiles(const std::string& path) {
    return getFiles(path, [](const boost::filesystem::path dirent)
                            { return (dirent.extension() == WillowProjector::cloud_extension &&
                              boost::starts_with(dirent.stem().string(), WillowProjector::cloud_prefix)); });
}

std::vector<boost::filesystem::path> getPoseFiles(const std::string& path, const std::string& prefix) {
    return getFiles(path, [prefix](const boost::filesystem::path dirent)
                            { return (dirent.extension() == WillowProjector::viewpoint_extension &&
                              boost::starts_with(dirent.stem().string(), prefix)); });
}

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

    return cloud_ptr;
}

Pose loadPose(const std::string& pose_path) {
    auto elems = loadNumericData<float>(pose_path);
    if ( elems.size() != 16 )
        throw std::runtime_error("No pose found in " + pose_path);

    Eigen::Matrix4f pose(elems.data());
    return pose.transpose();
}

}

WillowProjector::WillowProjector(const std::string& views_path) : views_path{views_path} {}

AlignedVector<View> WillowProjector::generateViews(const FullCloud::ConstPtr& /*cloud*/) const {
    auto cloud_files = getCloudFiles(views_path);

    auto views = AlignedVector<View>{};
    views.reserve(cloud_files.size());
    for (const auto& file : cloud_files) {
        const auto& idx_str = file.stem().string().substr(strlen(cloud_prefix));
        views.emplace_back(loadView(idx_str));
    }

    return views;
}

std::vector<int> WillowProjector::loadIndices(const std::string& indices_path) const {
    auto indices = loadNumericData<int>(indices_path);
    if ( indices.empty() )
        throw std::runtime_error("No indices found in " + indices_path);

    return indices;
}

View WillowProjector::loadView(const std::string& view_id) const {
    const auto cloud_path = views_path + '/' + cloud_prefix + view_id + cloud_extension;
    const auto indices_path = views_path + '/' + indices_prefix + view_id + indices_extension;
    const auto pose_path = views_path + '/' + viewpoint_prefix + view_id + viewpoint_extension;

    auto cloud = loadCloud(cloud_path);
    auto indices = boost::make_shared<std::vector<int>>(loadIndices(pose_path));
    auto viewpoint = loadPose(pose_path);

    pcl::ExtractIndices<FullPoint> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setNegative(true);
    extract.setKeepOrganized(true);

    extract.filter(*cloud);

    return View{ Image{cloud}, viewpoint };
}


WillowDatabase::WillowDatabase(const std::string& base_path) : base_path{base_path} {}

Model WillowDatabase::loadModel(const std::string& obj_name) const {
    const auto obj_path = base_path + '/' + obj_name;

    auto projector = descry::WillowProjector(obj_path + '/' + views_dir);
    auto full =  loadCloud(obj_path + '/' + full_cloud_filename);

    return Model{full, projector};
}

std::unordered_map<std::string, Model> WillowDatabase::loadDatabase() const {
    namespace fs = boost::filesystem;

    auto models = std::unordered_map<std::string, Model>{};
    for (auto dirent : boost::make_iterator_range(fs::directory_iterator(base_path),
                                                  fs::directory_iterator())) {
        if( dirent.status().type() == fs::directory_file ) {
            auto obj_name = dirent.path().string();
            models.emplace(obj_name, loadModel(obj_name));
        }
    }

    return models;
}


std::vector<WillowTestSet::AnnotatedScene>
WillowTestSet::loadTest(const std::string& scenes_path, const std::string& ground_truth_path) const {
    auto scene_files = getCloudFiles(scenes_path);

    auto annotated_scenes = std::vector<AnnotatedScene>{};
    annotated_scenes.reserve(scene_files.size());
    for (const auto& file : scene_files) {
        annotated_scenes.emplace_back(loadCloud(file.string()),
                                      loadInstances(ground_truth_path, file.stem().string()));
    }

    return annotated_scenes;
}

WillowTestSet::InstanceMap
WillowTestSet::loadInstances(const std::string& ground_truth_path, const std::string& scene_name) const {
    static constexpr auto obj_token = "object_";
    const auto pose_prefix = scene_name + '_' + obj_token;

    // ordered
    auto instances = InstanceMap{};

    auto pose_files = getPoseFiles(ground_truth_path, pose_prefix);
    for (const auto& file : pose_files) {
        const auto& object_name = file.stem().string().substr(pose_prefix.size() - strlen(obj_token),
                                                              pose_prefix.rfind('_'));

        auto pose = loadPose(file.string());

        auto instance = instances.find(object_name);
        if(instance == instances.end())
            instances.insert({object_name, {pose}});
        else
            instance->second.emplace_back(pose);
    }

    return instances;
}

}
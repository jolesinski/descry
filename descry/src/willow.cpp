#include <descry/willow.h>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/range/iterator_range.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>

namespace descry {

WillowProjector::WillowProjector(const std::string& views_path) : views_path{views_path} {}

AlignedVector<View> WillowProjector::generateViews(const FullCloud::ConstPtr& /*cloud*/) const {
    namespace fs = boost::filesystem;

    auto views = AlignedVector<View>{};
    for (auto dirent : boost::make_iterator_range(fs::directory_iterator(views_path),
                                                  fs::directory_iterator()))
    {
        const auto& dirent_path = dirent.path();
        if( dirent_path.extension() == cloud_extension )
        {
            const auto& idx_str = dirent_path.stem().string().substr(strlen(cloud_prefix));
            views.emplace_back(loadView(idx_str));
        }
    }

    return views;
}

template <class T>
std::vector<T> loadData(const std::string& path) {
    std::vector<T> elems;
    std::ifstream infile(path);

    if ( infile.bad() )
        throw std::runtime_error("Unable to load data from " + path);

    std::istream_iterator<T> eos;
    std::istream_iterator<T> iit(infile);

    std::copy(iit, eos, std::back_inserter(elems));

    return elems;
}

Pose WillowProjector::loadPose(const std::string& pose_path) const {
    auto elems = loadData<float>(pose_path);
    if ( elems.size() != 16 )
        throw std::runtime_error("No pose found in " + pose_path);

    Eigen::Matrix4f pose(elems.data());
    return pose.transpose();
}

std::vector<int> WillowProjector::loadIndices(const std::string& indices_path) const {
    auto indices = loadData<int>(indices_path);
    if ( indices.empty() )
        throw std::runtime_error("No indices found in " + indices_path);

    return indices;
}

namespace {
FullCloud::Ptr loadCloud(const std::string& cloud_path) {
    FullCloud::Ptr cloud_ptr(new FullCloud);
    if( pcl::io::loadPCDFile<pcl::PointXYZRGBA>(cloud_path, *cloud_ptr) == -1 )
        throw std::runtime_error("Unable to load pcd from" + cloud_path);

    return cloud_ptr;
}
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

std::vector<Model> WillowDatabase::loadDatabase() const {
    namespace fs = boost::filesystem;

    auto models = std::vector<Model>{};
    for (auto dirent : boost::make_iterator_range(fs::directory_iterator(base_path),
                                                  fs::directory_iterator())) {
        if( dirent.status().type() == fs::directory_file )
            models.emplace_back(loadModel(dirent.path().string()));
    }

    return models;
}

}
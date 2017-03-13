#ifndef DESCRY_WILLOW_H
#define DESCRY_WILLOW_H

#include <descry/model.h>

namespace descry {

/*
 * Load projections from Willow-like database
 */
class WillowProjector : public Projector {
public:
    static constexpr auto cloud_prefix = "cloud_";
    static constexpr auto cloud_extension = ".pcd";
    static constexpr auto indices_prefix = "object_indices_";
    static constexpr auto indices_extension = ".txt";
    static constexpr auto viewpoint_prefix = "pose_";
    static constexpr auto viewpoint_extension = ".txt";

    WillowProjector(const std::string& views_path);
    virtual ~WillowProjector() = default;
    AlignedVector<View> generateViews(const FullCloud::ConstPtr& cloud) const override;

    Pose loadPose(const std::string& pose_path) const;
    std::vector<int> loadIndices(const std::string& indices_path) const;
    View loadView(const std::string& view_id) const;
private:
    std::string views_path;
};

class WillowDatabase {
public:
    static constexpr auto views_dir = "views";
    static constexpr auto full_cloud_filename = "3D_model.pcd";

    WillowDatabase(const std::string& base_path);
    Model loadModel(const std::string& obj_name) const;
    std::vector<Model> loadDatabase() const;
private:
    std::string base_path;
};

}

#endif //DESCRY_WILLOW_H

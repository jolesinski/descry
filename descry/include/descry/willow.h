#ifndef DESCRY_WILLOW_H
#define DESCRY_WILLOW_H

#include <descry/model.h>
#include <unordered_map>

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
    std::unordered_map<std::string, Model> loadDatabase() const;
private:
    std::string base_path;
};

class WillowTestSet : public descry::WillowDatabase {
public:
    static constexpr auto scenes_dir = "willow_test_set";
    static constexpr auto models_dir = "willow_models";
    static constexpr auto gt_dir = "willow_annotations/willow";

    using InstanceMap = std::unordered_map<std::string, AlignedVector<Pose>>;
    using AnnotatedScene = std::pair<FullCloud::Ptr, InstanceMap>;

    WillowTestSet(const std::string& base_path) :
            WillowDatabase(base_path + '/' + models_dir),
            base_path_(base_path) {};

    std::vector<AnnotatedScene> loadSingleTest(const std::string& test_name) const {
        const auto scenes_path = base_path_ + '/' + scenes_dir + '/' + test_name;
        const auto gt_path = base_path_ + '/' + gt_dir + '/' + test_name;

        return loadTest(scenes_path, gt_path);
    }

    InstanceMap loadInstances(const std::string& ground_truth_path,
                              const std::string& scene_name) const;

    std::vector<AnnotatedScene> loadTest(const std::string& scenes_path,
                                         const std::string& gt_path) const;
private:

    std::string base_path_;
};

}

#endif //DESCRY_WILLOW_H

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
    WillowProjector(const Config& model_cfg);
    virtual ~WillowProjector() = default;
    AlignedVector<View> generateViews(const FullCloud::ConstPtr& cloud) const override;

    View loadView(const std::string& cloud_path,
                  const std::string& indices_path,
                  const std::string& pose_path) const;
private:
    const Config views_cfg_;
};

class WillowDatabase {
public:
    WillowDatabase(const Config& db_cfg);
    Model loadModel(const std::string& model_name) const;
    std::vector<std::string> getModelNames() const;
private:
    const Config db_cfg_;
};

class WillowTestSet {
public:
    using InstanceMap = std::unordered_map<std::string, AlignedVector<Pose>>;
    using AnnotatedScene = std::pair<FullCloud::Ptr, InstanceMap>;

    WillowTestSet(const Config& db_cfg) : test_set_cfg_(db_cfg["scenes"]) {};

    std::vector<std::string> getTestNames() const;

    template <typename... Args>
    std::vector<AnnotatedScene> loadSingleTest(const std::string& test_name, Args... args) const {
        return loadTest(test_set_cfg_[test_name], args...);
    }

    InstanceMap loadInstances(const Config& instances_cfg) const;
    std::vector<AnnotatedScene> loadTest(const Config& test_cfg, std::size_t max_frames) const;
    std::vector<AnnotatedScene> loadTest(const Config& test_cfg) const {
        return loadTest(test_cfg, test_cfg.size());
    }
private:

    const Config test_set_cfg_;
};

}

#endif //DESCRY_WILLOW_H

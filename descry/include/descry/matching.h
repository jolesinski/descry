#ifndef DESCRY_MATCHING_H
#define DESCRY_MATCHING_H

#include <descry/image.h>
#include <descry/config/matcher.h>
#include <pcl/correspondence.h>

// TODO:
// add thrust reduce min distance strategy

namespace descry {

template <class Descriptor>
class Matcher {
public:
    using DualDescriptors = cupcl::DualContainer<Descriptor>;

    class Strategy {
    public:
        virtual void setModel(const std::vector<DualDescriptors>& model) = 0;
        virtual std::vector<pcl::CorrespondencesPtr> match(const DualDescriptors& scene) = 0;
        virtual ~Strategy() = default;
    };

    bool configure(const Config& config);

    void setModel(const std::vector<DualDescriptors>& model);

    std::vector<pcl::CorrespondencesPtr> match(const DualDescriptors& scene);
private:
    std::unique_ptr<Strategy> strategy_ = nullptr;
};

}

#endif //DESCRY_MATCHING_H

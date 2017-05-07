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
    class Strategy {
    public:
        virtual void setModel(const std::vector<DescriptorContainer<Descriptor>>& model) = 0;
        virtual void setModel(std::vector<DescriptorContainer<Descriptor>>&& model) = 0;
        virtual std::vector<pcl::CorrespondencesPtr> match(const DescriptorContainer<Descriptor>& scene) = 0;
        virtual ~Strategy() = default;
    };

    bool configure(const Config& config);

    void setModel(const std::vector<DescriptorContainer<Descriptor>>& model);
    void setModel(std::vector<DescriptorContainer<Descriptor>>&& model);

    std::vector<pcl::CorrespondencesPtr> match(const DescriptorContainer<Descriptor>& scene);
private:
    std::unique_ptr<Strategy> strategy_ = nullptr;
};

}

#endif //DESCRY_MATCHING_H

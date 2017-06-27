#ifndef DESCRY_MATCHER_H
#define DESCRY_MATCHER_H

#include <descry/image.h>
#include <descry/descriptors.h>
#include <descry/config/matcher.h>
#include <pcl/correspondence.h>

// TODO:
// add thrust reduce min distance strategy
namespace descry {

template<class Descriptor>
class Matcher {
public:
    bool configure(const Config& config);

    virtual void train(const std::vector<Description<Descriptor>>& model);
    virtual void train(std::vector<Description<Descriptor>>&& model);
    virtual std::vector<pcl::CorrespondencesPtr> match(const Description<Descriptor>& scene);

    virtual ~Matcher() = default;
private:
    std::unique_ptr<Matcher<Descriptor>> strategy_ = nullptr;
};

}

#endif //DESCRY_MATCHER_H

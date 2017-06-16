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
    class Strategy {
    public:
        virtual void train(const std::vector<Description<Descriptor>>& model) = 0;

        virtual void train(std::vector<Description<Descriptor>>&& model) = 0;

        virtual std::vector<pcl::CorrespondencesPtr> match(const Description<Descriptor>& scene) = 0;

        virtual ~Strategy() = default;
    };

    bool configure(const Config& config);

    void train(const std::vector<Description<Descriptor>>& model);

    void train(std::vector<Description<Descriptor>>&& model);

    std::vector<pcl::CorrespondencesPtr> match(const Description<Descriptor>& scene);

private:
    std::unique_ptr<Strategy> strategy_ = nullptr;
};

}

#endif //DESCRY_MATCHER_H

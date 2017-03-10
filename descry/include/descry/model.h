
#ifndef DESCRY_MODEL_H
#define DESCRY_MODEL_H

#include <descry/common.h>
#include <descry/image.h>

namespace descry {

class Model
{
public:
    template<class Descriptor>
    using Description = cupcl::DualContainer<Descriptor>;

    struct View
    {
        Image image;
        Pose viewpoint;
        std::vector<bool> mask;
    };

    struct Projector
    {
        Perspective perspective;
        float sphere_radius;
        int sphere_turns;
        int sphere_divisions;

        View project(const FullCloud::ConstPtr& full, const Pose& viewpoint) const noexcept;
        std::vector<View> generateViews(const FullCloud::ConstPtr& cloud) const noexcept;
    };

    Model(const FullCloud::ConstPtr& full, const Projector& projector);
    Model(const FullCloud::ConstPtr& full, std::vector<View>&& views);

    const FullCloud::ConstPtr& getFullCloud() const { return full_cloud; }
    const std::vector<View>& getViews() const { return views; }
protected:
    FullCloud::ConstPtr full_cloud;
    std::vector<View> views;
};

}

#endif //DESCRY_MODEL_H

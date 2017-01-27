
#ifndef DESCRY_MODEL_H
#define DESCRY_MODEL_H

#include <descry/common.h>
#include <vector>

namespace descry {

class Model
{
public:
    struct View
    {
        using Vector = std::vector<View, Eigen::aligned_allocator<View>>;

        Pose viewpoint;
        Perspective perspective;
        PointCloud::Ptr cloud;
        Normals::Ptr normals;
        RFs::Ptr rfs;
        std::vector<bool> mask;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    };

    struct Projector
    {
        Perspective perspective;
        float sphere_radius;
        int sphere_turns;
        int sphere_divisions;

        View project(const PointCloud::ConstPtr& full, const Pose& viewpoint) const noexcept;
        View::Vector generateViews(const PointCloud::ConstPtr& cloud) const noexcept;
    };

    Model(const PointCloud::ConstPtr& full, const Projector& projector);
    Model(const PointCloud::ConstPtr& full, const View::Vector& views);

protected:
    PointCloud::ConstPtr full_cloud;
    View::Vector views;
};

class ModelInstance : public Model {
public:
    using Pose = Eigen::Matrix4f;

    ModelInstance(const Model& model, Pose pose);
protected:
    Pose pose;
};

}

#endif //DESCRY_MODEL_H

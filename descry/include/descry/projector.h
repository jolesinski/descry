#ifndef DESCRY_PROJECTOR_H
#define DESCRY_PROJECTOR_H

#include <descry/common.h>
#include <descry/image.h>
#include <descry/config/projection.h>

namespace descry {

struct View
{
    Image image;
    Pose viewpoint;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class Projector
{
public:
    virtual ~Projector() = default;
    virtual AlignedVector<View> generateViews(const FullCloud::ConstPtr& cloud) const = 0;
};

/*
 * Project views using spiral tesselation, see:
 * Abrate, Marco, and Fabrizio Pollastri. "Spiral tessellation on the sphere."
 * IAENG International Journal of Applied Mathematics 42.3 (2012): 129-134.
 */
class SphericalProjector : public Projector {
public:
    SphericalProjector(const Config& cfg, const Perspective& projection);
    virtual ~SphericalProjector() = default;
    AlignedVector<View> generateViews(const FullCloud::ConstPtr& cloud) const override;

    View project(const FullCloud::ConstPtr& full, const Pose& viewpoint) const;
private:
    float sphere_radius;
    int spiral_divisions;
    int spiral_turns;
    Perspective perspective;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif //DESCRY_PROJECTOR_H

#ifndef DESCRY_IMAGE_H
#define DESCRY_IMAGE_H

#include <descry/common.h>
#include <descry/cupcl/memory.h>

namespace descry {

class Image {
public:
    Image(FullCloud::ConstPtr cloud) : full(cloud) {};

    const DualConstFullCloud& getFullCloud() const;
    const DualPerpective& getProjection() const;
    const DualShapeCloud& getShapeCloud() const;
    const DualNormals& getNormals() const;

private:
    DualConstFullCloud full;
    mutable DualPerpective projection;
    mutable DualShapeCloud shape;
    DualNormals normals;
};

}

#endif //DESCRY_IMAGE_H

#ifndef DESCRY_IMAGE_H
#define DESCRY_IMAGE_H

#include <descry/common.h>

namespace descry {

class Image {
public:
    Image(FullCloud::ConstPtr cloud) : full(cloud) {};

    const DualConstFullCloud& getFullCloud() const;
    const DualPerpective& getProjection() const;
    const DualShapeCloud& getShapeCloud() const;

    const DualNormals& getNormals() const;
    void setNormals(DualNormals&& normals);

    const DualRefFrames& getRefFrames() const;
    void setRefFrames(DualRefFrames&& ref_frames);

    const DualShapeCloud& getShapeKeypoints() const;
    void setShapeKeypoints(DualShapeCloud&& keypoints);

private:
    DualConstFullCloud full;
    mutable DualPerpective projection;
    mutable DualShapeCloud shape;
    DualNormals normals;
    DualRefFrames ref_frames;
    DualShapeCloud shape_keypoints;
};

}

#endif //DESCRY_IMAGE_H

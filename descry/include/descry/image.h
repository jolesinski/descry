#ifndef DESCRY_IMAGE_H
#define DESCRY_IMAGE_H

#include <descry/common.h>

namespace descry {

class Image {
public:
    Image() = default;
    explicit Image(FullCloud::ConstPtr cloud) : full(cloud) {};

    const DualConstFullCloud& getFullCloud() const;
    const DualPerpective& getProjection() const;
    const DualShapeCloud& getShapeCloud() const;
    const cv::Mat& getColorMat() const;
    const cv::Mat& getDepthMat() const;

    const DualNormals& getNormals() const;
    void setNormals(DualNormals&& normals);

    bool empty() const noexcept { return full.empty(); }
private:
    DualConstFullCloud full;
    mutable DualPerpective projection;
    mutable DualShapeCloud shape;
    mutable cv::Mat color;
    mutable cv::Mat depth;
    DualNormals normals;
};

}

#endif //DESCRY_IMAGE_H

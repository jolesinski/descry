#include <descry/image.h>
#include <descry/cupcl/conversion.h>
#include <pcl/common/projection_matrix.h>

namespace {

descry::Perspective
computeProjection(const descry::FullCloud::ConstPtr& in, int pyramid = 5) {
    descry::Perspective projection;
    projection.setZero();

    const unsigned ySkip = (std::max)(in->height >> pyramid, unsigned(1));
    const unsigned xSkip = (std::max)(in->width >> pyramid, unsigned(1));

    std::vector<int> indices;
    indices.reserve(in->size() >> (pyramid << 1));

    for (unsigned yIdx = 0, idx = 0; yIdx < in->height; yIdx += ySkip, idx += in->width * ySkip)
        for (unsigned xIdx = 0, idx2 = idx; xIdx < in->width; xIdx += xSkip, idx2 += xSkip)
            indices.push_back(idx2);

    pcl::estimateProjectionMatrix<descry::FullPoint>(in, projection, indices);

    return projection;
}

}

namespace descry {

const DualConstFullCloud& Image::getFullCloud() const {
    return full;
}

const DualPerpective& Image::getProjection() const {
    if (projection.empty())
        projection.reset(std::make_unique<descry::Perspective>(computeProjection(full.host())));
    return projection;
}

const DualShapeCloud& Image::getShapeCloud() const {
    if (shape.empty())
        shape = cupcl::convertToXYZ(full);
    return shape;
}

const DualNormals& Image::getNormals() const {
    return normals;
}

void Image::setNormals(DualNormals&& normals) {
    this->normals = std::move(normals);
}

const DualRefFrames& Image::getRefFrames() const {
    return ref_frames;
}

void Image::setRefFrames(DualRefFrames&& ref_frames) {
    this->ref_frames = std::move(ref_frames);
}

const DualShapeCloud& Image::getShapeKeypoints() const {
    return shape_keypoints;
}

void Image::setShapeKeypoints(DualShapeCloud&& keypoints) {
    this->shape_keypoints = std::move(keypoints);
}

}
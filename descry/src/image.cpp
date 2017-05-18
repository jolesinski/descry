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

cv::Mat convertToColorMat(const descry::FullCloud::ConstPtr& in) {
    typedef cv::Point3_<uint8_t> Pixel;
    cv::Mat frame = cv::Mat::zeros(in->height, in->width, CV_8UC3);
    frame.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
        const auto& point = in->at(position[1], position[0]);
        pixel.x = point.b;
        pixel.y = point.g;
        pixel.z = point.r;
    });

    return frame;
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

const cv::Mat& Image::getColorMat() const {
    if (color.empty())
        color = convertToColorMat(full.host());

    return color;
}

const DualNormals& Image::getNormals() const {
    return normals;
}

void Image::setNormals(DualNormals&& normals) {
    this->normals = std::move(normals);
}

}
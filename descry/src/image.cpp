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

std::vector<cv::KeyPoint> convertShapeToColorKeypoints(const descry::ShapeCloud& keys,
                                                       const descry::Perspective& proj) {
    std::vector<cv::KeyPoint> cv_keys;

    for (auto key : keys.points) {
        Eigen::Vector4f vec = key.getArray4fMap();
        key.getArray3fMap() = proj * vec;
        cv_keys.emplace_back(key.x/key.z, key.y/key.z, -1);
    }

    return cv_keys;
}

descry::DualShapeCloud convertColorToShapeKeypoints(const std::vector<cv::KeyPoint>& cv_keys,
                                                    const descry::DualShapeCloud& full) {
    auto keys = descry::make_cloud<descry::ShapePoint>();

    keys->points.reserve(cv_keys.size());
    keys->width = static_cast<uint32_t>(cv_keys.size());
    keys->height = 1;

    for (auto key : cv_keys)
        keys->points.emplace_back(full.host()->at(static_cast<int>(key.pt.x), static_cast<int>(key.pt.y)));

    return descry::DualShapeCloud{std::move(keys)};
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

const DualRefFrames& Image::getRefFrames() const {
    return ref_frames;
}

void Image::setRefFrames(DualRefFrames&& ref_frames) {
    this->ref_frames = std::move(ref_frames);
}

const Image::Keypoints& Image::getKeypoints() const {
    keys.initPerspective(*getProjection().host());
    return keys;
}

const DualShapeCloud& Image::Keypoints::getShape() const {
    return shape;
}

DualShapeCloud& Image::Keypoints::getShape(const DualShapeCloud& full) {
    if (shape.empty() && !color.empty() && !full.empty())
        shape = convertColorToShapeKeypoints(color, full);

    return shape;
}

const std::vector<cv::KeyPoint>& Image::Keypoints::getColor() const {
    if (color.empty() && !shape.empty() && !projection.empty())
        color = convertShapeToColorKeypoints(*shape.host(), *projection.host());
    return color;
}

void Image::Keypoints::set(DualShapeCloud &&keypoints) {
    shape = std::move(keypoints);
}

void Image::Keypoints::set(std::vector<cv::KeyPoint> &&keypoints) {
    color = std::move(keypoints);
}

void Image::Keypoints::initPerspective(const descry::Perspective &proj) {
    if (projection.empty())
        projection.reset(std::make_unique<descry::Perspective>(proj));
}

}
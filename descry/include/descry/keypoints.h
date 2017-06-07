#ifndef DESCRY_KEYPOINTS_H
#define DESCRY_KEYPOINTS_H

#include <descry/config/keypoints.h>
#include <descry/image.h>
#include <descry/viewer.h>

namespace descry {

class Keypoints {
public:
    Keypoints() = default;
    explicit Keypoints(const Image& image) {
        init(*image.getProjection().host(), image.getShapeCloud().host());
    }
    explicit Keypoints(DualShapeCloud&& keypoints, const Image& image) : shape(std::move(keypoints)) {
        init(*image.getProjection().host(), image.getShapeCloud().host());
    }
    explicit Keypoints(std::vector<cv::KeyPoint>&& keypoints, const Image& image) : color(std::move(keypoints)) {
        init(*image.getProjection().host(), image.getShapeCloud().host());
    }

    Keypoints(const Keypoints& other) : color(other.color), full(other.full) {
        if (!other.shape.empty())
            shape = DualShapeCloud{other.shape.host()};
        if (!other.projection.empty())
            projection = DualPerpective{std::make_unique<Perspective>(*other.projection.host())};
    }

    Keypoints& operator=(const Keypoints& other) {
        color = other.color;
        full = other.full;
        if (!other.shape.empty())
            shape = DualShapeCloud{other.shape.host()};
        if (!other.projection.empty())
            projection = DualPerpective{std::make_unique<Perspective>(*other.projection.host())};

        return *this;
    }

    Keypoints(Keypoints&& other) = default;
    Keypoints& operator=(Keypoints&& other) = default;

    std::size_t size() const noexcept { return color.empty() ? shape.size() : color.size(); }
    bool empty() const noexcept { return size() == 0; }

    void set(DualShapeCloud&& keypoints) {
        shape = std::move(keypoints);
    }

    void set(std::vector<cv::KeyPoint>&& keypoints) {
        color = std::move(keypoints);
    }

    void init(const descry::Perspective& proj, const descry::ShapeCloud::ConstPtr& full) {
        if (projection.empty())
            projection.reset(std::make_unique<descry::Perspective>(proj));

        this->full = full;
    }

    const DualShapeCloud& getShape() const {
        if (shape.empty() && !color.empty() && full != nullptr && !full->empty())
            shape = color_to_shape(color, *full);
        return shape;
    }

    const std::vector<cv::KeyPoint>& getColor() const {
        if (color.empty() && !shape.empty() && !projection.empty())
            color = shape_to_color(*shape.host(), *projection.host());
        return color;
    }

    static std::vector<cv::KeyPoint> shape_to_color(const descry::ShapeCloud& keys, const descry::Perspective& proj) {
        std::vector<cv::KeyPoint> cv_keys;

        for (auto key : keys.points) {
            Eigen::Vector4f vec = key.getArray4fMap();
            key.getArray3fMap() = proj * vec;
            cv_keys.emplace_back(key.x/key.z, key.y/key.z, -1);
        }

        return cv_keys;
    }

    static descry::DualShapeCloud color_to_shape(const std::vector<cv::KeyPoint>& cv_keys, const descry::ShapeCloud& full) {
        auto keys = descry::make_cloud<descry::ShapePoint>();

        keys->points.reserve(cv_keys.size());
        keys->width = static_cast<uint32_t>(cv_keys.size());
        keys->height = 1;

        for (auto key : cv_keys) {
            auto x = static_cast<unsigned int>(key.pt.x);
            auto y = static_cast<unsigned int>(key.pt.y);
            keys->points.emplace_back(full.at(x, y));
        }

        return descry::DualShapeCloud{std::move(keys)};
    }

private:
    mutable DualShapeCloud shape;
    mutable std::vector<cv::KeyPoint> color;

    DualPerpective projection;
    ShapeCloud::ConstPtr full;
};

class KeypointDetector {
public:
    bool configure(const Config &config);
    bool is_configured() const noexcept;

    Keypoints compute(const Image &image) const;

private:
    std::function<Keypoints( const Image& )> nest_;
    Viewer<Keypoints> viewer_;
};

}

#endif //DESCRY_KEYPOINTS_H

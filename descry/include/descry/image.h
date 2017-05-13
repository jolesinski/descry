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
    const cv::Mat& getColorMat() const;

    const DualNormals& getNormals() const;
    void setNormals(DualNormals&& normals);

    const DualRefFrames& getRefFrames() const;
    void setRefFrames(DualRefFrames&& ref_frames);

    class Keypoints {
    public:
        void set(DualShapeCloud&& keypoints);
        void set(std::vector<cv::KeyPoint>&& keypoints);

        void init(const descry::Perspective& proj, const descry::ShapeCloud::ConstPtr& full);

        const DualShapeCloud& getShape() const;
        const std::vector<cv::KeyPoint>& getColor() const;
    private:
        mutable DualShapeCloud shape;
        mutable std::vector<cv::KeyPoint> color;

        DualPerpective projection;
        ShapeCloud::ConstPtr full;
    };

    const Keypoints& getKeypoints() const;

    void setKeypoints(Keypoints&& keypoints) {
        keys = std::move(keypoints);
    }

    template <class KeyType>
    void setKeypoints(KeyType&& keypoints) {
        keys.set(std::forward<KeyType>(keypoints));
    }

private:
    DualConstFullCloud full;
    mutable DualPerpective projection;
    mutable DualShapeCloud shape;
    mutable cv::Mat color;
    DualNormals normals;
    DualRefFrames ref_frames;
    mutable Keypoints keys;
};

}

#endif //DESCRY_IMAGE_H

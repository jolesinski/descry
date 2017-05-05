#include <algorithm>
#include <iostream>

#include <descry/keypoints.h>
#include <descry/normals.h>
#include <descry/willow.h>
#include <descry/test/config.h>
#include <descry/test/data.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>

std::vector<int> gt_filter(const descry::FullCloud::ConstPtr& model,
                           const descry::Pose& pose,
                           const descry::ShapeKeypoints& keys, double rad) {
    std::vector<int> model_keys;

    pcl::PointXYZRGBA min_p, max_p;
    pcl::getMinMax3D(*model, min_p, max_p);
    min_p.getVector4fMap() = pose * min_p.getVector4fMap() - Eigen::Vector4f::Constant(rad);
    max_p.getVector4fMap() = pose * max_p.getVector4fMap() + Eigen::Vector4f::Constant(rad);
    std::cout << "min " << min_p << " max " << max_p << std::endl;

    int idx = 0;
    for (auto key : keys.host()->points) {
        if (key.x < min_p.x || key.y < min_p.y || key.z < min_p.z ||
                key.x > max_p.x || key.y > max_p.y || key.z > max_p.z)
            continue;

        model_keys.emplace_back(idx);

        ++idx;
    }

    std::cout << idx << std::endl;

    return model_keys;
}

void eval(const descry::Model& model,
          const descry::ShapeKeypoints& scene,
          const descry::AlignedVector<descry::Pose>& gt, double rad) {

    auto gt_indices = gt_filter(model.getFullCloud(), gt.front(), scene, rad);

    unsigned int model_keys = 0;
    unsigned int matches = 0;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(scene.host());

    auto match_cloud = descry::make_cloud<pcl::PointXYZ>();

    for (const auto& view : model.getViews()) {
        model_keys += view.image.getShapeKeypoints().size();
        for (auto key : view.image.getShapeKeypoints().host()->points) {
            std::vector<float> dist;
            std::vector<int> ind;
            key.getVector4fMap() = gt.front() * view.viewpoint * key.getVector4fMap();
            matches += kdtree.radiusSearch(key, rad, ind, dist, 1);
            if (!ind.empty()) {
                match_cloud->push_back(scene.host()->at(ind.front()));
            }
        }
    }


    std::cout << "Model keys " << model_keys << " scene gt " << gt_indices.size() << " all " << scene.size() << " matches " << matches << " in rad " << rad <<std::endl;

//    descry::FullCloud::Ptr transformed(new descry::FullCloud);
//    pcl::transformPointCloud (*model.getFullCloud(), *transformed, gt.front());
//    //pcl::transformPointCloud (*model.getFullCloud(), *transformed, gt.front() * model.viewpoint);
//
//    pcl::visualization::CloudViewer viewer("Cloud Viewer");
//    viewer.showCloud(transformed, "model");
//    viewer.showCloud(scene.host(), "scene");
//    while (!viewer.wasStopped ())
//        continue;


    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
    viewer.setBackgroundColor (0, 0, 0);

    viewer.addPointCloud(scene.host(), "scene");
    viewer.addPointCloud(match_cloud, "matches");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "matches");
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, 255, 0 , 0, "matches");

    viewer.addCoordinateSystem (.3);
    viewer.initCameraParameters ();

    viewer.spin ();
}

float eval_gt_distance(const descry::Config& cfg) {
    auto scene_cfg = cfg["scene"];
    auto model_cfg = cfg["model"];
    auto eval_rad = cfg["metrics"]["radius"].as<double>();

    std::cout << "Keys config\n" << scene_cfg << std::endl;

    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    auto test_name = scene_cfg["name"].as<std::string>();
    auto model_name = model_cfg["name"].as<std::string>();

    // load data
    auto test_data = willow.loadSingleTest(test_name);
    auto model = willow.loadModel(model_name);
    auto image = descry::Image(test_data.front().first);

    std::cout << "Data loaded" << std::endl;

    auto gt_handle = std::find_if(std::begin(test_data.front().second), std::end(test_data.front().second),
                                  [model_name](const auto& el){ return el.first == model_name; });
    if (gt_handle == std::end(test_data.front().second))
        DESCRY_THROW(descry::InvalidConfigException, "Model not present in scene");

    std::cout << "Model " << model_name << " found in " << test_name << std::endl;

    auto kdet = descry::ShapeKeypointDetector{};
    if(!kdet.configure(scene_cfg["keypoints"]))
        DESCRY_THROW(descry::InvalidConfigException, "Invalid keypoints config");
    std::cout << "Keypoints configured" << std::endl;

    // prepare
    auto prep = descry::Preprocess{};
    if(!prep.configure(model_cfg))
        DESCRY_THROW(descry::InvalidConfigException, "Invalid preprocessing config");
    std::cout << "Preprocess configured" << std::endl;

    // compute
    model.prepare(prep);
    auto keys = kdet.compute(image);

    eval(model, keys, gt_handle->second, eval_rad);

    return 0.0f;
}

cv::Mat get_color_mat(const pcl::PointCloud<pcl::PointXYZRGBA>& cloud) {
    using Pixel = cv::Point3_<uint8_t>;
    cv::Mat frame = cv::Mat::zeros(cloud.height, cloud.width, CV_8UC3);
    frame.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
        const auto& point = cloud.at(position[1], position[0]);
        pixel.x = point.b;
        pixel.y = point.g;
        pixel.z = point.r;
    });

    return frame;
}

cv::Mat get_color_mat(const descry::Image& image) {
    return get_color_mat(*image.getFullCloud().host());
}

std::vector<cv::KeyPoint> get_cv_keys(const descry::Image& image) {
    auto keys = image.getShapeKeypoints().host();

    std::vector<cv::KeyPoint> kp;
    for (auto key : keys->points) {
        Eigen::Vector4f vec = key.getArray4fMap();
        key.getArray3fMap() = (*image.getProjection().host()) * vec;
        kp.emplace_back(key.x/key.z, key.y/key.z, -1);
    }

    return kp;
}

void view_keys(const descry::Image& image, std::string window_name) {
    //auto keys = compute_harris_3d(image);
    //auto keys = compute_iss_3d(image);

    //copy to opencv mat
    auto frame = get_color_mat(image);
    auto keys = get_cv_keys(image);

    std::cout << "Detected " << keys.size() << std::endl;

    cv::drawKeypoints(frame, keys, frame, cv::Scalar(0,0,128), cv::DrawMatchesFlags::DEFAULT);

    cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
    cv::imshow( window_name, frame );
}

void view_keys(const descry::Model& model, std::string window_name) {
    //auto keys = compute_harris_3d(image);
    //auto keys = compute_iss_3d(image);

    //copy to opencv mat
    for (const auto& view : model.getViews()) {
        auto frame = get_color_mat(view.image);
        auto keys = get_cv_keys(view.image);

        std::cout << "Detected " << keys.size() << std::endl;
        view_keys(view.image, "model");
        cv::waitKey(100);
    }
}

void compute_keys(const descry::Config& cfg) {
    auto scene_cfg = cfg["scene"];
    auto model_cfg = cfg["model"];
    auto willow = descry::WillowTestSet(descry::test::loadDBConfig());
    auto test_name = scene_cfg["name"].as<std::string>();
    auto model_name = model_cfg["name"].as<std::string>();


    auto test_data = willow.loadSingleTest(test_name, 1);
    auto image = descry::Image(test_data.front().first);

    auto nest = descry::NormalEstimation{};
    nest.configure(cfg["scene"][descry::config::normals::NODE_NAME]);
    auto kdet = descry::ShapeKeypointDetector{};

    auto keys_cfg = cfg["scene"][descry::config::keypoints::NODE_NAME];
    kdet.configure(keys_cfg);

    image.setNormals(nest.compute(image));
    image.setShapeKeypoints(kdet.compute(image));
    view_keys(image, "scene");

    // load
    auto model = willow.loadModel(model_name);
    std::cout << "Preprocess configured" << std::endl;

    // prepare
    auto prep = descry::Preprocess{};
    if(!prep.configure(model_cfg))
        DESCRY_THROW(descry::InvalidConfigException, "Invalid model preprocessing config");

    // compute
    model.prepare(prep);
    view_keys(model, "model");
}

void compute_color_keys() {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    auto image = descry::Image(descry::test::loadSceneCloud());
    const auto& full = image.getFullCloud().host();

    //copy to opencv mat
    typedef cv::Point3_<uint8_t> Pixel;
    cv::Mat frame = cv::Mat::zeros(full->width, full->height, CV_8UC3);
    frame.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
        const auto& point = full->at(full->width - position[0] - 1, position[1]);
        pixel.x = point.b;
        pixel.y = point.g;
        pixel.z = point.r;
    });

    // compute orb
    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    std::cout << "Max Features " << orb->getMaxFeatures() << std::endl;
    std::cout << "Edge thresh " << orb->getEdgeThreshold() << std::endl;
    std::cout << "Fast thresh " << orb->getFastThreshold() << std::endl;
    std::cout << "First level " << orb->getFirstLevel() << std::endl;
    std::cout << "N Levels " << orb->getNLevels() << std::endl;
    std::cout << "Patch size " << orb->getPatchSize() << std::endl;
    std::cout << "Scale factor " << orb->getScaleFactor() << std::endl;
    std::cout << "Score type " << orb->getScoreType() << std::endl;
    std::cout << "WTA K " << orb->getWTA_K() << std::endl;

    orb->setMaxFeatures(1000);

    orb->detectAndCompute(frame, cv::noArray(), kp, desc);

    std::cout << "Detected " << kp.size() << std::endl;

    cv::drawKeypoints(frame, kp, frame, cv::Scalar(0,0,128), cv::DrawMatchesFlags::DEFAULT);

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    cv::imshow( "Display window", frame );
    cv::waitKey(0);
}

int main(int argc, char * argv[]) {
    //TODO: pick best pose view from model or merge all
    // calculate keypoints, extract gt keys from scene keys (FULL CLOUD needed)
    // calculate how many gt keys has a model match within given precision distance

    // parse config
    descry::Config cfg;
    try {
        if (argc > 1)
            cfg = YAML::LoadFile(argv[1]);
    } catch (...) { }

    if (cfg.IsNull()) {
        std::cerr << "No config provided" << std::endl;
        return EXIT_FAILURE;
    }

    compute_keys(cfg);
    compute_color_keys();
    //auto eval_keys = eval_gt_distance(cfg);

    return EXIT_SUCCESS;
}

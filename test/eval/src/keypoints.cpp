#include <algorithm>
#include <iostream>

#include <descry/keypoints.h>
#include <descry/willow.h>
#include <descry/test/config.h>
#include <descry/test/data.h>

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

class KeypointEvaluator {
public:
    //descry::Model get_model(const descry::Config& params);

    float operator()(const descry::Config& params);
};

float KeypointEvaluator::operator()(const descry::Config& cfg) {
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

int main(int argc, char * argv[]) {
    //TODO: pick best pose view from model or merge all
    // calculate keypoints, extract gt keys from scene keys (FULL CLOUD needed)
    // calculate how many gt keys has a model match within given precision distance

    // parse config
    descry::Config cfg;
    if (argc > 1) {
        cfg = YAML::LoadFile(argv[1]);
    } else {
        std::cerr << "No config provided" << std::endl;
        return -1;
    }

    auto eval_keys = KeypointEvaluator();

    return EXIT_SUCCESS;
}

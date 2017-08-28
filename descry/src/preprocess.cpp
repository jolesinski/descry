#include <descry/preprocess.h>

#include <descry/keypoints.h>
#include <descry/normals.h>
#include <descry/ref_frames.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>

#include <descry/logger.h>
#include <descry/config/common.h>
#include <descry/latency.h>
#include <descry/utils.h>

namespace descry {

void Preprocess::configure(const Config& cfg) {
    if (!cfg.IsMap())
        return;

    log_latency_ = cfg[config::LOG_LATENCY].as<bool>(false);
    viewer_.configure(cfg);

    auto& passthrough_cfg = cfg[config::preprocess::PASSTHROUGH];
    if (passthrough_cfg) {
        pcl::PassThrough<descry::FullPoint> pass;
        pass.setKeepOrganized(true);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.0, passthrough_cfg.as<float>());
        steps_.emplace_back( [pass, this] (Image& image) mutable {
            auto latency = measure_scope_latency("Passthrough", this->log_latency_);
            auto filtered = make_cloud<descry::FullPoint>();
            pass.setInputCloud(image.getFullCloud().host());
            pass.filter(*filtered);
            remove_indices(*filtered, *pass.getRemovedIndices());
            image = Image{filtered};
            viewer_.show(image);
        });
    }

    auto& smoothing_cfg = cfg[config::preprocess::SMOOTHING];
    if (smoothing_cfg) {
        pcl::FastBilateralFilterOMP<descry::FullPoint> smooth;
        smooth.setNumberOfThreads(smoothing_cfg[config::THREADS].as<unsigned int>(0));
        smooth.setSigmaR(smoothing_cfg[config::preprocess::SIGMA_R].as<float>(0.05f));
        smooth.setSigmaS(smoothing_cfg[config::preprocess::SIGMA_S].as<float>(15.0f));
        steps_.emplace_back( [smooth, this] (Image& image) mutable {
            auto latency = measure_scope_latency("Smoothing", this->log_latency_);
            auto filtered = make_cloud<descry::FullPoint>();
            smooth.setInputCloud(image.getFullCloud().host());
            smooth.applyFilter(*filtered);
            image = Image{filtered};
            viewer_.show(image);
        });
    }

    auto& normals_cfg = cfg[config::normals::NODE_NAME];
    if (normals_cfg) {
        auto nest = descry::NormalEstimation{};

        if (nest.configure(normals_cfg)) {
            steps_.emplace_back( [nest{std::move(nest)}, this] (Image& image)
                                 {
                                     auto latency = measure_scope_latency("Normals", this->log_latency_);
                                     image.setNormals(nest.compute(image));
                                 }
            );
        }
    }

    auto& segmentation_cfg = cfg[config::preprocess::SEGMENTATION];
    if (segmentation_cfg) {
        pcl::OrganizedMultiPlaneSegmentation<descry::FullPoint, pcl::Normal, pcl::Label> mps;
        mps.setMinInliers(segmentation_cfg[config::preprocess::MIN_INLIERS].as<unsigned int>(10000));
        mps.setProjectPoints(segmentation_cfg[config::preprocess::PROJECT_POINTS].as<bool>(true));
        mps.setAngularThreshold(segmentation_cfg[config::preprocess::ANGULAR_THRESH].as<double>(0.017453 * 3));
        mps.setDistanceThreshold(segmentation_cfg[config::preprocess::DISTANCE_THRESH].as<double>(0.02));
        steps_.emplace_back( [mps, this] (Image& image) mutable {
            auto latency = measure_scope_latency("Plane segmentation", this->log_latency_);
            auto model_coefficients = std::vector<pcl::ModelCoefficients>{};
            auto inlier_indices = std::vector<pcl::PointIndices>{};
            mps.setInputNormals(image.getNormals().host());
            mps.setInputCloud(image.getFullCloud().host());

            auto planar_regions = AlignedVector<pcl::PlanarRegion<FullPoint>>{};
            auto labels = make_cloud<pcl::Label>();
            auto label_indices = std::vector<pcl::PointIndices>{};
            auto boundary_indices = std::vector<pcl::PointIndices>{};
            mps.segmentAndRefine(planar_regions, model_coefficients, inlier_indices,
                                 labels, label_indices, boundary_indices);

            auto filtered = make_cloud<descry::FullPoint>();
            pcl::copyPointCloud(*image.getFullCloud().host(),*filtered);
            for (auto& indices : inlier_indices)
                remove_indices(*filtered, indices.indices);

            if (!filtered->empty()) {
                auto normals = image.getNormals().host();
                image = Image{filtered};
                image.setNormals(std::move(normals));
            }
            viewer_.show(image);
        });
    }
}

void Preprocess::process(Image& image) const {
    for(auto& step : steps_)
        step(image);
}

}

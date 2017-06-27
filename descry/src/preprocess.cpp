#include <descry/preprocess.h>

#include <descry/keypoints.h>
#include <descry/normals.h>
#include <descry/ref_frames.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/fast_bilateral_omp.h>

#include <pcl/segmentation/planar_region.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/organized_connected_component_segmentation.h>

#include <descry/logger.h>
#include <descry/config/common.h>
#include <pcl/filters/extract_indices.h>

namespace descry {

bool Preprocess::configure(const Config& cfg) {
    auto& passthrough_cfg = cfg[config::preprocess::PASSTHROUGH];
    if (passthrough_cfg) {
        pcl::PassThrough<descry::FullPoint> pass;
        pass.setKeepOrganized(true);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.0, passthrough_cfg.as<float>());
        steps_.emplace_back( [pass] (Image& image) mutable {
            auto filtered = make_cloud<descry::FullPoint>();
            pass.setInputCloud(image.getFullCloud().host());
            pass.filter(*filtered);
            image = Image{filtered};
        });
    }

    auto& smoothing_cfg = cfg[config::preprocess::SMOOTHING];
    if (smoothing_cfg) {
        pcl::FastBilateralFilterOMP<descry::FullPoint> smooth;
        smooth.setNumberOfThreads(smoothing_cfg[config::THREADS].as<unsigned int>(0));
        smooth.setSigmaR(smoothing_cfg[config::preprocess::SIGMA_R].as<float>(0.05f));
        smooth.setSigmaS(smoothing_cfg[config::preprocess::SIGMA_S].as<float>(15.0f));
        steps_.emplace_back( [smooth] (Image& image) mutable {
            auto filtered = make_cloud<descry::FullPoint>();
            smooth.setInputCloud(image.getFullCloud().host());
            smooth.applyFilter(*filtered);
            image = Image{filtered};
        });
    }

    auto& normals_cfg = cfg[config::normals::NODE_NAME];
    if (normals_cfg) {
        auto nest = descry::NormalEstimation{};

        if(!nest.configure(normals_cfg))
            return false;

        steps_.emplace_back( [nest{std::move(nest)}]
                (Image& image) { image.setNormals(nest.compute(image)); }
        );
    }

    auto& segmentation_cfg = cfg[config::preprocess::SEGMENTATION];
    if (segmentation_cfg) {
        pcl::OrganizedMultiPlaneSegmentation<descry::FullPoint, pcl::Normal, pcl::Label> mps;
        mps.setMinInliers(segmentation_cfg[config::preprocess::MIN_INLIERS].as<unsigned int>(10000));
        mps.setAngularThreshold(0.017453 * 2.0); //3 degrees
        mps.setDistanceThreshold(0.02); //2cm
        steps_.emplace_back( [mps] (Image& image) mutable {
            auto model_coefficients = std::vector<pcl::ModelCoefficients>{};
            auto inlier_indices = std::vector<pcl::PointIndices>{};
            mps.setInputNormals(image.getNormals().host());
            mps.setInputCloud(image.getFullCloud().host());
            mps.segment(model_coefficients, inlier_indices);

            auto filtered = make_cloud<descry::FullPoint>();
            pcl::ExtractIndices<descry::FullPoint> extract;
            for (auto& indices : inlier_indices) {
                auto indices_ptr = pcl::PointIndicesPtr{&indices};
                if (filtered->empty())
                    extract.setInputCloud(image.getFullCloud().host());
                else
                    extract.setInputCloud(filtered);
                extract.setIndices(indices_ptr);
                extract.setNegative(false);
                extract.setKeepOrganized(true);
                extract.filter(*filtered);
            }

            image = Image{filtered};
        });
    }

    return true;
}

void Preprocess::process(Image& image) const {
    for(auto& step : steps_)
        step(image);
}

}

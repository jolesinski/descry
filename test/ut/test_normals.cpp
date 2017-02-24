#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/normals.h>

// TODO: add config files

TEST_CASE( "Configuring normals", "[normals]" ) {
    auto nest = descry::NormalEstimation{};
    auto cfg = YAML::Load("");

    REQUIRE(!nest.configure(cfg));

    cfg.SetTag("normals");
    cfg["type"] = descry::config::normals::OMP_TYPE;

    REQUIRE(nest.configure(cfg));

    cfg.reset(descry::test::loadFullConfig());

    REQUIRE(!nest.configure(cfg));
    REQUIRE(nest.configure(cfg["recognizer"]["normals"]));
}

TEST_CASE( "Normal estimation on planar cloud", "[normals]") {
    auto nest = descry::NormalEstimation{};

    SECTION("OMP") {
        REQUIRE(nest.configure(descry::test::normals::loadConfigOmp()));
    }
    SECTION("INT") {
        REQUIRE(nest.configure(descry::test::normals::loadConfigInt()));
    }
    SECTION("CUPCL") {
        REQUIRE(nest.configure(descry::test::normals::loadConfigCupcl()));
    }

    Eigen::Vector4f coeffs;
    coeffs << 1, 1, -1, 1;
    auto image = descry::Image(descry::test::createPlanarCloud(128, 128, coeffs));

    auto normals = nest.compute(image);
    REQUIRE(!normals.empty());
    REQUIRE(normals.size() == image.getFullCloud().size());

    auto ground_truth = coeffs.head<3>();
    ground_truth.normalize();

    auto nan_count = 0u;
    auto accuracy = 1e-3;
    auto& h_normals = normals.host();
    for(int i = 0; i < h_normals->width; i += 8) {
        for(int j = 0; j < h_normals->height; j += 8) {
            const pcl::Normal& normal = h_normals->at(i, j);
            if (!pcl::isFinite(normal) && ++nan_count)
                continue;

            CAPTURE( i + j * h_normals->width );
            REQUIRE( normal.normal_x == Approx(ground_truth(0)).epsilon( accuracy ) );
            REQUIRE( normal.normal_y == Approx(ground_truth(1)).epsilon( accuracy ) );
            REQUIRE( normal.normal_z == Approx(ground_truth(2)).epsilon( accuracy ) );
            REQUIRE( normal.curvature == Approx(0).epsilon(1e-3) );
        }
    }

    if(nan_count > 0)
        WARN("Invalid normals: " << nan_count << "/" << h_normals->size());
}

TEST_CASE( "Normal estimation on real cloud", "[normals]") {
    auto gt_nest = descry::NormalEstimation{};
    gt_nest.configure(descry::test::normals::loadConfigOmp());

    auto nest = descry::NormalEstimation{};
    SECTION("INT") {
        REQUIRE(nest.configure(descry::test::normals::loadConfigInt()));
    }
    SECTION("CUPCL") {
        REQUIRE(nest.configure(descry::test::normals::loadConfigCupcl()));
    }

    auto image = descry::Image(descry::test::loadSceneCloud());
    auto gt_normals = gt_nest.compute(image);
    auto normals = nest.compute(image);
    REQUIRE(!gt_normals.empty());
    REQUIRE(!normals.empty());
    REQUIRE(gt_normals.size() == image.getFullCloud().size());
    REQUIRE(normals.size() == image.getFullCloud().size());

    const auto max_angle = 1;
    const auto max_curv_dif = 1;
    auto nan_count = 0u;
    auto angle_count = 0u;
    auto curv_count = 0u;
    auto& h_gt_normals = gt_normals.host();
    auto& h_normals = normals.host();
    for(int i = 0; i < h_normals->width; i += 16) {
        for(int j = 0; j < h_normals->height; j += 16) {
            const pcl::Normal& gt_normal = h_gt_normals->at(i, j);
            const pcl::Normal& normal = h_normals->at(i, j);

            if (!pcl::isFinite(gt_normal) || !pcl::isFinite(normal)) {
                ++nan_count;
                continue;
            }

            auto angle = acosf(gt_normal.normal_x * normal.normal_x
                                  + gt_normal.normal_y * normal.normal_y
                                  + gt_normal.normal_z * normal.normal_z);
            if (angle > max_angle)
                ++angle_count;

            if (fabsf(gt_normal.curvature - normal.curvature) > max_curv_dif)
                ++curv_count;
        }
    }

    REQUIRE(nan_count < 300);
    REQUIRE(angle_count < 100);
    REQUIRE(curv_count < 100);

}
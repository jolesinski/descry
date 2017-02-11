#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/test/config.h>
#include <descry/normals.h>

TEST_CASE( "Configuring normals", "[normals]" ) {
    auto nest = descry::NormalEstimation{};
    auto cfg = YAML::Load("");

    REQUIRE(!nest.configure(cfg));

    cfg.SetTag("normals");
    cfg["type"] = "omp";

    REQUIRE(nest.configure(cfg));

    cfg.reset(descry::test::loadFullConfig());

    REQUIRE(!nest.configure(cfg));
    REQUIRE(nest.configure(cfg["sparse"]["normals"]));
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
    auto plane = descry::Image(descry::test::createPlanarCloud(128, 128, coeffs));

    auto normals = nest.compute(plane);
    REQUIRE(!normals->empty());
    REQUIRE(normals->points.size() == plane.getFullCloud().getSize());

    auto ground_truth = coeffs.head<3>();
    ground_truth.normalize();

    auto nan_count = 0u;
    auto accuracy = 1e-3;
    for(int i = 0; i < 100/*normals->width*/; i += 8) {
        for(int j = 0; j <  10/*normals->height*/; j += 8) {
            const pcl::Normal& normal = normals->at(i, j);
            if (!pcl::isFinite(normal) && ++nan_count)
                continue;

            CAPTURE( i + j * normals->width );
            REQUIRE( normal.normal_x == Approx(ground_truth(0)).epsilon( accuracy ) );
            REQUIRE( normal.normal_y == Approx(ground_truth(1)).epsilon( accuracy ) );
            REQUIRE( normal.normal_z == Approx(ground_truth(2)).epsilon( accuracy ) );
            REQUIRE( normal.curvature == Approx(0).epsilon(1e-3) );
        }
    }

    if(nan_count > 0)
        WARN("Invalid normals: " << nan_count << "/" << normals->size());
}
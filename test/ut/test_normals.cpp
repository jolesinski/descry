#include <catch.hpp>

#include <descry/test/data.h>
#include <descry/normals.h>

TEST_CASE( "Configuring normals", "[normals]" ) {
    auto nest = descry::NormalEstimation{};
    auto cfg = YAML::Load("");

    REQUIRE(!nest.configure(cfg));

    cfg.SetTag("normals");
    cfg["type"] = "omp";

    REQUIRE(nest.configure(cfg));

    cfg.reset(YAML::LoadFile(descry::test::CONFIG_PATH));

    REQUIRE(!nest.configure(cfg));
    REQUIRE(nest.configure(cfg["sparse"]["normals"]));
}

TEST_CASE( "Normal estimation on planar cloud", "[normals]") {
    auto nest = descry::NormalEstimation{};
    auto cfg = YAML::LoadFile(descry::test::CONFIG_PATH)["sparse"]["normals"];

    REQUIRE(nest.configure(cfg));

    Eigen::Vector4f coeffs;
    coeffs << 1, 1, -1, 1;
    auto plane = descry::test::createPlanarCloud(128, 128, coeffs);

    auto normals = nest.compute(plane);
    REQUIRE(!normals->empty());

    auto ground_truth = coeffs.head<3>();
    ground_truth.normalize();

    auto accuracy = 5e-4;
    for(int i = 0; i < normals->width; i += 8)
    {
        for(int j = 0; j < normals->height; j += 8)
        {
            const pcl::Normal& normal = normals->at(i, j);

            CAPTURE( i + j * normals->width );
            REQUIRE( normal.normal_x == Approx(ground_truth(0)).epsilon( accuracy ) );
            REQUIRE( normal.normal_y == Approx(ground_truth(1)).epsilon( accuracy ) );
            REQUIRE( normal.normal_z == Approx(ground_truth(2)).epsilon( accuracy ) );
            REQUIRE( normal.curvature == Approx(0).epsilon(1e-3) );
        }
    }
}
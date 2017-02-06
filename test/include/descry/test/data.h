
#ifndef DESCRY_TEST_DATA_H
#define DESCRY_TEST_DATA_H

#include <descry/common.h>
#include <pcl/io/pcd_io.h>
#include <descry/test/data_paths.h>

namespace descry {
namespace test {

    inline PointCloud::Ptr loadCloudPCD(const std::string &path) {
        PointCloud::Ptr cloud(new PointCloud());

        if (pcl::io::loadPCDFile<Point>(path, *cloud) == -1) {
            PCL_ERROR ("Couldn't read file %s \n", path.c_str());
            exit(-1);
        }

        return cloud;
    }

    inline PointCloud::Ptr loadSceneCloud() { return loadCloudPCD(SCENE_PATH); }
    inline PointCloud::Ptr loadModelCloud() { return loadCloudPCD(MODEL_PATH); }

    static constexpr auto VGA_WIDTH = 640u;
    static constexpr auto VGA_HEIGHT = 480u;

    inline PointCloud::Ptr
    createPlanarCloud(unsigned width, unsigned height, const Eigen::Vector4f coeffs)
    {
        assert(coeffs(2) != 0);

        PointCloud::Ptr cloud(new PointCloud);
        cloud->width = width;
        cloud->height = height;
        cloud->points.resize(cloud->width * cloud->height);

        // Fill cloud with plane
        for(int y = 0; y < cloud->height; ++y)
        {
            for(int x = 0; x < cloud->width; ++x)
            {
                Point& point = cloud->at(x, y);
                point.x = x;
                point.y = y;
                point.z = (coeffs(0) * x + coeffs(1) * y + coeffs(3))/-coeffs(2);
            }
        }

        return cloud;
    }
}
}

#endif //DESCRY_TEST_DATA_H

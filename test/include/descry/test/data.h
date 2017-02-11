
#ifndef DESCRY_TEST_DATA_H
#define DESCRY_TEST_DATA_H

#include <descry/common.h>
#include <pcl/io/pcd_io.h>
#include <descry/test/data_paths.h>

namespace descry {
namespace test {

    inline FullCloud::Ptr loadCloudPCD(const std::string &path) {
        FullCloud::Ptr cloud(new FullCloud{});

        if (pcl::io::loadPCDFile<FullPoint>(path, *cloud) == -1) {
            PCL_ERROR ("Couldn't read file %s \n", path.c_str());
            exit(-1);
        }

        return cloud;
    }

    inline FullCloud::Ptr loadSceneCloud() { return loadCloudPCD(SCENE_PATH); }
    inline FullCloud::Ptr loadModelCloud() { return loadCloudPCD(MODEL_PATH); }

    static constexpr auto VGA_WIDTH = 640u;
    static constexpr auto VGA_HEIGHT = 480u;

    inline FullCloud::Ptr
    createPlanarCloud(unsigned width, unsigned height, const Eigen::Vector4f coeffs)
    {
        assert(coeffs(2) != 0);

        FullCloud::Ptr cloud(new FullCloud{});
        cloud->width = width;
        cloud->height = height;
        cloud->points.resize(cloud->width * cloud->height);

        // Fill cloud with plane
        for(int y = 0; y < cloud->height; ++y)
        {
            for(int x = 0; x < cloud->width; ++x)
            {
                FullPoint& point = cloud->at(x, y);
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

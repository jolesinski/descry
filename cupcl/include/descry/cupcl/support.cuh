
#ifndef DESCRY_CUPCL_RADIUS_CUH
#define DESCRY_CUPCL_RADIUS_CUH

#include <pcl/point_types.h>
#include <host_defines.h>

// This can easily fill up stack memory
#define MAX_NN_SIZE 256//128
#define MIN_COV_NN   10

__host__ __device__ int
radiusSearch(const int query_idx,
             const pcl::PointXYZ* in,
             const int width, const int height,
             const float* projection,
             const float radius,
             int* indices,
             float* sqr_distances = nullptr);

__host__ __device__ bool
getSupportCovariance(const int query_idx,
                     const pcl::PointXYZ* in,
                     const int width, const int height,
                     const float* projection,
                     const float radius,
                     Eigen::Matrix3f& covariance);

__host__ __device__ void
getWeightedCovariance(const int query_idx,
                      const pcl::PointXYZ* in,
                      const int* nn_indices,
                      const float* nn_sqr_dist,
                      const int nn_size,
                      const float radius,
                      Eigen::Matrix3f& covariance);

#endif //DESCRY_CUPCL_RADIUS_CUH

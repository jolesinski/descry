
#ifndef CUPCL_UTILS_H
#define CUPCL_UTILS_H

#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <host_defines.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

__host__ __device__ inline Eigen::Vector3f
getVector3f(const pcl::PointXYZ& point)
{
  Eigen::Vector3f vector;
  vector << point.x, point.y, point.z;
  return vector;
}

__host__ __device__ inline Eigen::Vector4f
getVector4f(const pcl::PointXYZ& point)
{
  Eigen::Vector4f vector;
  vector << point.x, point.y, point.z, 1;
  return vector;
}

__device__ inline void
zeroNaNs(pcl::PointXYZ& point)
{
  if(!isfinite(point.x) ||
     !isfinite(point.y) ||
     !isfinite(point.z))
  {
    point.x = 0;
    point.y = 0;
    point.z = 0;
  }
}

#define CudaSyncAndBail() __cudaSyncAndBail( __FILE__, __LINE__ )

inline void __cudaSyncAndBail( const char *file, const int line )
{
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    std::cerr << "cudaCheckError() failed at "
              << file << ":" << line << " : " << cudaGetErrorString(err) << std::endl;
    exit( -1 );
  }

  err = cudaDeviceSynchronize();
  if( cudaSuccess != err )
  {
    std::cerr << "cudaCheckError() with sync failed at "
              << file << ":" << line << " : " << cudaGetErrorString(err) << std::endl;
    exit( -1 );
  }
}

#endif //CUPCL_UTILS_H

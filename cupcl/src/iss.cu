#include <descry/cupcl/iss.h>
#include <descry/cupcl/support.cuh>
#include <descry/cupcl/eigen.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <descry/cupcl/device_utils.h>

__global__ void
computeSaliencies(const cupcl::PointT* in,
                  const int width, const int height,
                  const float* projection,
                  const float salient_rad,
                  const float eps1,
                  const float eps2,
                  const float eps3,
                  float* out)
{
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = width*v + u;

  Eigen::Matrix3f scatter = Eigen::Matrix3f::Identity();

  if (!isfinite(in[idx].x) ||
      !getSupportCovariance(idx, in, width, height, projection, salient_rad, scatter))
    return;

  Eigen::Vector3f lambdas;
  eigenvalsSymm3x3(scatter, lambdas);

  if( lambdas(1) / lambdas(0) < eps1
      && lambdas(2) / lambdas(1) < eps2
      && lambdas(2) > eps3 )
  {
    out[idx] = lambdas(2);
  }
}

__global__ void
computeNoNMaxima(const cupcl::PointT* in,
                 const int width, const int height,
                 const float* projection,
                 const float* saliency,
                 const float nonmax_rad,
                 const int min_nn,
                 int* is_keypoint)
{
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = width*v + u;

  const cupcl::PointT& query = in[idx];

  auto query_l3 = saliency[idx];
  if (query_l3 == 0 || !isfinite(query.x))
    return;

  int nn_indices[MAX_NN_SIZE];
  int nn_size = radiusSearch(idx, in, width, height,
                                projection, nonmax_rad, nn_indices);

  if (nn_size >= min_nn)
  {
    bool is_max = true;

    for (int nn_idx = 0 ; nn_idx < nn_size; ++nn_idx)
      if( query_l3 < saliency[nn_indices[nn_idx]] )
      {
        is_max = false;
        break;
      }

    is_keypoint[idx] = static_cast<int>(is_max);
  }
}

template<typename T>
struct is_true
{
  __host__ __device__
  bool operator()(const T& x)
  {
    return x != 0;
  }
};

void
cudaComputeISS(const cupcl::Cloud<cupcl::PointT>& cloud,
               const float resolution,
               const float eps1,
               const float eps2,
               const float eps3,
               cupcl::PointVecT& keys)
{
  float salient_rad = 6*resolution;
  float nonmax_rad = 4*resolution;
  int min_neighs = 5;

  thrust::device_vector<float> d_l3(cloud.getDeviceThrust()->size(), 0);
  float* d_l3_array = thrust::raw_pointer_cast(&d_l3[0]);

  thrust::device_vector<int> d_is_keypoint(cloud.getDeviceThrust()->size(), 0);
  int* d_is_keypoint_array = thrust::raw_pointer_cast(&d_is_keypoint[0]);

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks(cloud.getPCL()->width / threadsPerBlock.x,
                 cloud.getPCL()->height / threadsPerBlock.y);

  computeSaliencies<< < numBlocks, threadsPerBlock >> >
      (cloud.getDeviceRaw(), cloud.getPCL()->width, cloud.getPCL()->height,
          cloud.getDeviceProjectionMatrix(), salient_rad, eps1, eps2, eps3, d_l3_array);

  CudaSyncAndBail();

  computeNoNMaxima<< < numBlocks, threadsPerBlock >> >
      (cloud.getDeviceRaw(), cloud.getPCL()->width, cloud.getPCL()->height,
          cloud.getDeviceProjectionMatrix(), d_l3_array, nonmax_rad, min_neighs, d_is_keypoint_array);

  CudaSyncAndBail();

  thrust::device_vector<cupcl::PointT> d_keypoints;
  d_keypoints.reserve(cloud.getPCL()->size());
  auto key_end = thrust::copy_if(cloud.getDeviceThrust()->begin(), cloud.getDeviceThrust()->end(),
                                 d_is_keypoint.begin(), d_keypoints.begin(), is_true<float>());
  //int sum = thrust::reduce(thrust::device, d_is_keypoint.begin(), d_is_keypoint.end());

  size_t key_size = key_end - d_keypoints.begin();
  std::cout << "Sum " << key_size << std::endl;
  keys.resize(key_size);
  thrust::copy(d_keypoints.begin(), key_end, keys.begin());
}

cupcl::CloudPtrT
cupcl::computeISS(const cupcl::Cloud<cupcl::PointT>& cloud,
                  const float resolution,
                  const float eps1,
                  const float eps2,
                  const float eps3)
{
  cupcl::CloudPtrT keys (new cupcl::CloudT ());

  cudaComputeISS(cloud, resolution, eps1, eps2, eps3, keys->points);

  return keys;
}
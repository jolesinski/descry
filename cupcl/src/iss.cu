#include <descry/cupcl/iss.h>
#include <descry/cupcl/support.cuh>
#include <descry/cupcl/eigen.cuh>
#include <descry/cupcl/utils.cuh>
#include <descry/cupcl/unique.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/count.h>

__global__ void
computeSaliencies(const pcl::PointXYZ* in,
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
      && lambdas(2) > eps3 ) {
    out[idx] = lambdas(2);
  }
}

__global__ void
computeNoNMaxima(const pcl::PointXYZ* in,
                 const int width, const int height,
                 const float* projection,
                 const float* saliency,
                 const float nonmax_rad,
                 const int min_nn,
                 bool* is_keypoint)
{
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = width*v + u;

  const pcl::PointXYZ& query = in[idx];

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

    is_keypoint[idx] = is_max;
  }
}

namespace descry { namespace cupcl {

DualShapeCloud computeISS(const DualShapeCloud& points,
                          const DualPerpective& projection,
                          const ISSConfig& cfg) {
    assert(!points.empty());
    assert(!projection.empty());

    auto& d_points = points.device();
    const auto d_points_raw = d_points->getRaw();

    auto width = d_points->getWidth();
    auto height = d_points->getHeight();

    const auto d_projection_raw = projection.device()->getRaw();

    float salient_rad = cfg.salient_rad*cfg.resolution;
    float nonmax_rad = cfg.non_max_rad*cfg.resolution;
    int min_neighs = cfg.min_neighs;

    thrust::device_vector<float> d_l3(points.size(), 0);
    float* d_l3_array = thrust::raw_pointer_cast(&d_l3[0]);

    thrust::device_vector<bool> d_is_keypoint(points.size(), 0);
    bool* d_is_keypoint_array = thrust::raw_pointer_cast(&d_is_keypoint[0]);

    // FIXME: magic 32
    // FIXME: investigate alignment error, probably when conversion called from nvcc
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

    computeSaliencies<<< numBlocks, threadsPerBlock >>>(d_points_raw, width, height,
            d_projection_raw, salient_rad, cfg.lambda_ratio_21,
            cfg.lambda_ratio_32, cfg.lambda_threshold_3, d_l3_array);

    CudaSyncAndBail();

    computeNoNMaxima<<< numBlocks, threadsPerBlock >>>(d_points_raw, width, height,
            d_projection_raw, d_l3_array, nonmax_rad, min_neighs, d_is_keypoint_array);

    CudaSyncAndBail();

    using namespace thrust::placeholders;

    auto key_count = thrust::count_if(d_is_keypoint.begin(), d_is_keypoint.end(), _1);

    auto d_keys = std::make_unique<DeviceVector2d<pcl::PointXYZ>>(key_count, 1);
    auto key_end = thrust::copy_if(d_points->getThrust().begin(), d_points->getThrust().end(),
                                   d_is_keypoint.begin(), d_keys->getThrust().begin(), _1);

    size_t key_size = key_end - d_keys->getThrust().begin();

    return DualShapeCloud{std::move(d_keys)};
}

} }
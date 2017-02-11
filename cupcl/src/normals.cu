#include <descry/cupcl/utils.cuh>
#include <descry/cupcl/eigen.cuh>
#include <descry/cupcl/support.cuh>
#include <descry/cupcl/memory.h>
#include <descry/cupcl/unique.h>
#include <descry/common.h>

#include <device_launch_parameters.h>

#include <thrust/device_vector.h>
#include <pcl/point_types.h>

namespace {

__host__ __device__ void
fitPlane(const int query_idx, const pcl::PointXYZ *in,
         const int width, const int height,
         const float *projection,
         const float radius, pcl::Normal &normal) {
    Eigen::Matrix3f scatter = Eigen::Matrix3f::Identity();
    if (!isfinite(in[query_idx].x) ||
        !getSupportCovariance(query_idx, in, width, height, projection, radius, scatter)) {
        normal.normal_x = NAN;
        normal.normal_y = NAN;
        normal.normal_z = NAN;
        return;
    }

    Eigen::Vector3f normal_vec, lambdas;
    eigenvalsSymm3x3(scatter, lambdas);
    eigenvecSymm3x3(scatter, lambdas, 2, normal_vec);

    if (lambdas.nonZeros())
        normal.curvature = lambdas(2) / lambdas.sum();
    else
        normal.curvature = NAN;

    // orient towards viewpoint
    if (normal_vec.dot(getVector3f(in[query_idx])) > 0)
        normal_vec = -normal_vec;

    normal.normal_x = normal_vec(0);
    normal.normal_y = normal_vec(1);
    normal.normal_z = normal_vec(2);
}

__global__ void
computeNormalsKernel(const pcl::PointXYZ *in,
                     const int width, const int height,
                     const float *projection,
                     const float radius,
                     pcl::Normal *out) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = width * v + u;

    fitPlane(idx, in, width, height, projection, radius, out[idx]);
}

}

namespace descry { namespace cupcl {

DualNormals computeNormals(const DualShapeCloud& points,
                           const DualPerpective& projection,
                           const float radius) {
    assert(!points.empty());
    assert(!projection.empty());

    auto d_normals = std::make_unique<thrust::device_vector<pcl::Normal>>(points.getSize());
    pcl::Normal* d_norms_raw = thrust::raw_pointer_cast(&(*d_normals)[0]);
    const auto d_points_raw = thrust::raw_pointer_cast(&(*points.device())[0]);
    const auto d_projection_raw = thrust::raw_pointer_cast(&(*projection.device())[0]);

    dim3 threadsPerBlock(32, 32);

    // FIXME: pass width and height along with device container
    // FIXME: investigate alignment error, probably when conversion called from nvcc
    dim3 numBlocks(/*points.host()->width*/ 640 / threadsPerBlock.x,
                   /*points.host()->height*/ 480 / threadsPerBlock.y);

    computeNormalsKernel<< < numBlocks, threadsPerBlock >> >
                (d_points_raw, /*points.host()->width*/ 640, /*points.host()->height*/ 480,
                    d_projection_raw, radius, d_norms_raw);

    CudaSyncAndBail();

    return DualContainer<pcl::Normal>(std::move(d_normals));
}

}}
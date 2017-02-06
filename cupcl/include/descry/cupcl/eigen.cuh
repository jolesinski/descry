#ifndef DESCRY_CUPCL_EIGEN_H
#define DESCRY_CUPCL_EIGEN_H

#include <Eigen/Eigen>
#include <host_defines.h>

__host__ __device__ inline float
determinant3x3(const Eigen::Matrix3f& A) {
  return A(0,0)*(A(1,1)*A(2,2) - A(1,2)*A(2,1)) -
         A(0,1)*(A(1,0)*A(2,2) - A(1,2)*A(2,0)) +
         A(0,2)*(A(1,0)*A(2,1) - A(1,1)*A(2,0));
}

__host__ __device__ void
eigenvalsSymm3x3(const Eigen::Matrix3f& A, Eigen::Vector3f& lambdas);

__host__ __device__ void
eigenvecSymm3x3(const Eigen::Matrix3f& A, const Eigen::Vector3f& lambda,
                const int query_idx, Eigen::Vector3f& v);

__host__ __device__ void
eigenvecSymm3x3ForMinEigenval(const Eigen::Matrix3f& A, Eigen::Vector3f& v);

#endif //DESCRY_CUPCL_EIGEN_H

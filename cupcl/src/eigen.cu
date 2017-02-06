#include <descry/cupcl/eigen.cuh>

__host__ __device__ void
eigenvalsSymm3x3(const Eigen::Matrix3f& A, Eigen::Vector3f& lambdas) {
  auto p1 = A(0, 1)*A(0, 1) + A(0, 2)*A(0, 2) + A(1, 2)*A(1, 2);

  auto trace = A.trace();
  auto q = trace/3;
  auto p2 = (A(0,0) - q)*(A(0,0) - q) + (A(1,1) - q)*(A(1,1) - q) +
            (A(2,2) - q)*(A(2,2) - q) + 2 * p1;
  auto p = sqrt(p2 / 6);
  Eigen::Matrix3f B = (1 / p) * (A - q * Eigen::Matrix3f::Identity());
  auto r = determinant3x3(B) / 2;

  // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
  // but computation error can leave it slightly outside this range.
  float phi = 0;
  if (r <= -1)
    phi = M_PI / 3;
  else if (r < 1)
    phi = acos(r) / 3;

  // the eigenvalues satisfy eig3 <= eig2 <= eig1
  lambdas.coeffRef(0) = q + 2 * p * cos(phi);
  lambdas.coeffRef(2) = q + 2 * p * cos(phi + (2*M_PI/3));
  lambdas.coeffRef(1) = trace - lambdas.coeff(0) - lambdas.coeff(2); // since trace(A) = eig1 + eig2 + eig3
}

//works only for distinct eigenvals
__host__ __device__ void
eigenvecSymm3x3(const Eigen::Matrix3f& A, const Eigen::Vector3f& lambda,
                const int query_idx, Eigen::Vector3f& v) {
  Eigen::Vector3f lambdas;
  eigenvalsSymm3x3(A, lambdas);

  int idx_l1 = 0;
  int idx_l2 = 1;
  if (query_idx == 0)
  {
    idx_l1 = 1;
    idx_l2 = 2;
  }
  else if (query_idx == 1)
  {
    idx_l1 = 0;
    idx_l2 = 2;
  }

  Eigen::Matrix3f Al12 = (A - lambdas(idx_l1)*Eigen::Matrix3f::Identity()) *
                         (A - lambdas(idx_l2)*Eigen::Matrix3f::Identity());

  if( Al12.col(0).nonZeros() )
    v = Al12.col(0);
  else if( Al12.col(1).nonZeros() )
    v = Al12.col(1);
  else
    v = Al12.col(2);

  if( v.nonZeros() )
    v.normalize();
  else
    v.setConstant(NAN);
}

//works only for distinct eigenvals
__host__ __device__ void
eigenvecSymm3x3ForMinEigenval(const Eigen::Matrix3f& A, Eigen::Vector3f& v) {
  Eigen::Vector3f lambdas;
  eigenvalsSymm3x3(A, lambdas);
  eigenvecSymm3x3(A, lambdas, 2, v);
}
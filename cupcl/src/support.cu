#include <descry/cupcl/utils.cuh>
#include <descry/cupcl/support.cuh>
#include <math_functions.h>

// min and max indexes are inclusive
__host__ __device__ void
projectRadiusToSearchBox(const pcl::PointXYZ& query,
                         const int width, const int height,
                         const float radius,
                         const float* projection,
                         int& min_u, int& max_u,
                         int& min_v, int& max_v)
{
  float p0 = projection[0] * query.x + projection[1] * query.y
             + projection[2] * query.z + projection[3];
  float p1 = projection[4] * query.x + projection[5] * query.y
             + projection[6] * query.z + projection[7];
  float p2 = projection[8] * query.x + projection[9] * query.y
             + projection[10] * query.z + projection[11];

  float proj_min_u = (p0 - projection[0] * radius) / (p2 - projection[8] * radius);
  float proj_max_u = (p0 + projection[0] * radius) / (p2 + projection[8] * radius);
  float proj_min_v = (p1 - projection[5] * radius) / (p2 - projection[9] * radius);
  float proj_max_v = (p1 + projection[5] * radius) / (p2 + projection[9] * radius);

  min_u = static_cast<int>(floor(fmin(proj_min_u, proj_max_u)));
  max_u = static_cast<int>(ceil(fmax(proj_min_u, proj_max_u)));
  min_v = static_cast<int>(floor(fmin(proj_min_v, proj_max_v)));
  max_v = static_cast<int>(ceil(fmax(proj_min_v, proj_max_v)));
}

__host__ __device__ inline float
getSquareDistance(const pcl::PointXYZ& p1, const pcl::PointXYZ& p2)
{
  if (!isfinite(p1.x) || !isfinite(p2.x))
    return INFINITY;

  const float point_dist_x = p2.x - p1.x;
  const float point_dist_y = p2.y - p1.y;
  const float point_dist_z = p2.z - p1.z;

  return (point_dist_x * point_dist_x +
          point_dist_y * point_dist_y +
          point_dist_z * point_dist_z);
}

struct SpiralIndex
{
  SpiralIndex(int u0, int v0) : u(u0), v(v0) {}

  __host__ __device__ inline void
  next()
  {
    ++idx;

    // take step
    u += dir_u;
    v += dir_v;
    ++segment_passed;

    // change direction
    if (segment_passed == segment_length) {
      segment_passed = 0;

      int tmp = dir_u;
      dir_u = -dir_v;
      dir_v = tmp;

      if (dir_v == 0)
        ++segment_length;
    }
  }

  __host__ __device__ inline bool
  isWithinBounds(int width, int height) const
  {
    return u >= 0 && u < width && v >= 0 && v < height;
  }

  int idx = 0;
  int u, v;
  int dir_u = 1;
  int dir_v = 0;
  int segment_passed = 0;
  int segment_length = 1;
};

__host__ __device__ int
radiusSearch(const int query_idx,
             const pcl::PointXYZ* in,
             const int width, const int height,
             const float* projection,
             const float radius,
             int* indices,
             float* sqr_distances)
{
  if (!isfinite(in[query_idx].x))
    return 0;

  int min_u, max_u, min_v, max_v;
  projectRadiusToSearchBox(in[query_idx], width, height, radius,
                           projection, min_u, max_u, min_v, max_v);

  float sqr_radius = radius*radius;
  int nn_size = 0;
  int search_box_size = (max_u - min_u)*(max_v - min_v);

  int query_u = query_idx % width;
  int query_v = query_idx / width;


  for (SpiralIndex spiral(query_u, query_v);
       spiral.idx < search_box_size && nn_size < MAX_NN_SIZE;
       spiral.next())
  {
    if(!spiral.isWithinBounds(width, height))
      continue;

    int nn_idx = width*spiral.v + spiral.u;
    if( !isfinite(in[nn_idx].x) )
      continue;
    
    float sqr_distance = getSquareDistance(in[query_idx], in[nn_idx]);
    // check distance and add to results
    if (sqr_distance <= sqr_radius)
    {
      indices[nn_size] = nn_idx;
      if(sqr_distances != nullptr)
        sqr_distances[nn_size] = sqr_distance;
      nn_size++;
    }
  }

  return nn_size;
}

__host__ __device__ bool
getSupportCovariance(const int query_idx,
                     const pcl::PointXYZ* in,
                     const int width, const int height,
                     const float* projection,
                     const float radius,
                     Eigen::Matrix3f& covariance)
{
  int nn_indices[MAX_NN_SIZE] = {0};
  int nn_size = radiusSearch(query_idx, in, width, height,
                             projection, radius, nn_indices);

  if (nn_size < MIN_COV_NN)
    return false;

  covariance = Eigen::Matrix3f::Zero();
  for (int n_idx = 0; n_idx < nn_size; n_idx++)
  {
    const pcl::PointXYZ& neigh = in[nn_indices[n_idx]];
    float neigh_point[3] = {neigh.x, neigh.y, neigh.z};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        covariance.coeffRef(i * 3 + j) += ((neigh_point[i] - in[query_idx].data[i]) *
                          (neigh_point[j] - in[query_idx].data[j]))/(nn_size - 1);
  }
  return true;
}

__host__ __device__ void
getWeightedCovariance(const int query_idx,
                      const pcl::PointXYZ* in,
                      const int* nn_indices,
                      const float* nn_sqr_dist,
                      const int nn_size,
                      const float radius,
                      Eigen::Matrix3f& covariance)
{
  covariance = Eigen::Matrix3f::Zero();
  float summed_proximity = 0;
  for (int n_idx = 0; n_idx < nn_size; n_idx++)
  {
    const pcl::PointXYZ& neigh = in[nn_indices[n_idx]];
    float neigh_point[3] = {neigh.x, neigh.y, neigh.z};
    float proximity = radius - sqrtf(nn_sqr_dist[n_idx]);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        covariance.coeffRef(i * 3 + j) += ((neigh_point[i] - in[query_idx].data[i]) *
                                           (neigh_point[j] - in[query_idx].data[j])) * proximity;
    summed_proximity += proximity;
  }
  covariance /= summed_proximity;
}
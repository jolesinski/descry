#pragma once

/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: cvfh.h 4936 2012-03-07 11:12:45Z aaldoma $
 *
 */

#include <pcl/features/feature.h>
#include <pcl/search/pcl_search.h>
#include <pcl/common/common.h>

namespace pcl_fixed
{
    using namespace pcl;

    /** \brief OURCVFHEstimation2 estimates the Oriented, Unique and Repetable Clustered Viewpoint Feature Histogram (CVFH) descriptor for a given
     * point cloud dataset given XYZ data and normals, as presented in:
     *     - OUR-CVFH – Oriented, Unique and Repeatable Clustered Viewpoint Feature Histogram for Object Recognition and 6DOF Pose Estimation
     *     A. Aldoma, F. Tombari, R.B. Rusu and M. Vincze
     *     DAGM-OAGM 2012
     *     Graz, Austria
     * The suggested PointOutT is pcl::VFHSignature308.
     *
     * \author Aitor Aldoma
     * \ingroup features
     */
    template<typename PointInT, typename PointNT, typename PointOutT = pcl::VFHSignature308>
    class OURCVFHEstimation2 : public FeatureFromNormals<PointInT, PointNT, PointOutT>
    {
    public:
        typedef boost::shared_ptr<OURCVFHEstimation2<PointInT, PointNT, PointOutT> > Ptr;
        typedef boost::shared_ptr<const OURCVFHEstimation2<PointInT, PointNT, PointOutT> > ConstPtr;
        using Feature<PointInT, PointOutT>::feature_name_;
        using Feature<PointInT, PointOutT>::getClassName;
        using Feature<PointInT, PointOutT>::indices_;
        using Feature<PointInT, PointOutT>::k_;
        using Feature<PointInT, PointOutT>::search_radius_;
        using Feature<PointInT, PointOutT>::surface_;
        using FeatureFromNormals<PointInT, PointNT, PointOutT>::normals_;

        typedef typename Feature<PointInT, PointOutT>::PointCloudOut PointCloudOut;
        typedef typename pcl::search::Search<PointNormal>::Ptr KdTreePtr;
        typedef typename pcl::PointCloud<PointInT>::Ptr PointInTPtr;
        /** \brief Empty constructor. */
        OURCVFHEstimation2 () :
                vpx_ (0), vpy_ (0), vpz_ (0), leaf_size_ (0.005f), normalize_bins_ (false), curv_threshold_ (0.03f), cluster_tolerance_ (leaf_size_ * 3),
                eps_angle_threshold_ (0.125f), min_points_ (50), radius_normals_ (leaf_size_ * 3), centroids_dominant_orientations_ (),
                dominant_normals_ ()
        {
            search_radius_ = 0;
            k_ = 1;
            feature_name_ = "OURCVFHEstimation2";
            refine_clusters_ = 1.f;
            min_axis_value_ = 0.925f;
            axis_ratio_ = 0.8f;
        }
        ;

        /** \brief Creates an affine transformation from the RF axes
         * \param[in] evx the x-axis
         * \param[in] evy the y-axis
         * \param[in] evz the z-axis
         * \param[out] transformPC the resulting transformation
         * \param[in] center_mat 4x4 matrix concatenated to the resulting transformation
         */
        inline Eigen::Matrix4f
        createTransFromAxes (Eigen::Vector3f & evx, Eigen::Vector3f & evy, Eigen::Vector3f & evz, Eigen::Affine3f & transformPC,
                             Eigen::Matrix4f & center_mat)
        {
            Eigen::Matrix4f trans;
            trans.setIdentity (4, 4);
            trans (0, 0) = evx (0, 0);
            trans (1, 0) = evx (1, 0);
            trans (2, 0) = evx (2, 0);
            trans (0, 1) = evy (0, 0);
            trans (1, 1) = evy (1, 0);
            trans (2, 1) = evy (2, 0);
            trans (0, 2) = evz (0, 0);
            trans (1, 2) = evz (1, 0);
            trans (2, 2) = evz (2, 0);

            Eigen::Matrix4f homMatrix = Eigen::Matrix4f ();
            homMatrix.setIdentity (4, 4);
            homMatrix = transformPC.matrix ();

            Eigen::Matrix4f trans_copy = trans.inverse ();
            trans = trans_copy * center_mat * homMatrix;
            return trans;
        }

        /** \brief Computes SGURF and the shape distribution based on the selected SGURF
         * \param[in] processed the input cloud
         * \param[out] output the resulting signature
         * \param[in] cluster_indices the indices of the stable cluster
         */
        void
        computeRFAndShapeDistribution (PointInTPtr & processed, PointCloudOut &output, std::vector<pcl::PointIndices> & cluster_indices);

        /** \brief Computes SGURF
         * \param[in] centroid the centroid of the cluster
         * \param[in] normal_centroid the average of the normals
         * \param[in] processed the input cloud
         * \param[out] transformations the transformations aligning the cloud to the SGURF axes
         * \param[out] grid the cloud transformed internally
         * \param[in] indices the indices of the stable cluster
         */
        bool
        sgurf (Eigen::Vector3f & centroid, Eigen::Vector3f & normal_centroid, PointInTPtr & processed, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & transformations,
               PointInTPtr & grid, pcl::PointIndices & indices);

        /** \brief Removes normals with high curvature caused by real edges or noisy data
         * \param[in] cloud pointcloud to be filtered
         * \param[in] indices_to_use
         * \param[out] indices_out the indices of the points with higher curvature than threshold
         * \param[out] indices_in the indices of the remaining points after filtering
         * \param[in] threshold threshold value for curvature
         */
        void
        filterNormalsWithHighCurvature (const pcl::PointCloud<PointNT> & cloud, std::vector<int> & indices_to_use, std::vector<int> &indices_out,
                                        std::vector<int> &indices_in, float threshold);

        /** \brief Set the viewpoint.
         * \param[in] vpx the X coordinate of the viewpoint
         * \param[in] vpy the Y coordinate of the viewpoint
         * \param[in] vpz the Z coordinate of the viewpoint
         */
        inline void
        setViewPoint (float vpx, float vpy, float vpz)
        {
            vpx_ = vpx;
            vpy_ = vpy;
            vpz_ = vpz;
        }

        /** \brief Set the radius used to compute normals
         * \param[in] radius_normals the radius
         */
        inline void
        setRadiusNormals (float radius_normals)
        {
            radius_normals_ = radius_normals;
        }

        /** \brief Get the viewpoint.
         * \param[out] vpx the X coordinate of the viewpoint
         * \param[out] vpy the Y coordinate of the viewpoint
         * \param[out] vpz the Z coordinate of the viewpoint
         */
        inline void
        getViewPoint (float &vpx, float &vpy, float &vpz)
        {
            vpx = vpx_;
            vpy = vpy_;
            vpz = vpz_;
        }

        /** \brief Get the centroids used to compute different CVFH descriptors
         * \param[out] centroids vector to hold the centroids
         */
        inline void
        getCentroidClusters (std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > & centroids)
        {
            for (size_t i = 0; i < centroids_dominant_orientations_.size (); ++i)
                centroids.push_back (centroids_dominant_orientations_[i]);
        }

        /** \brief Get the normal centroids used to compute different CVFH descriptors
         * \param[out] centroids vector to hold the normal centroids
         */
        inline void
        getCentroidNormalClusters (std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > & centroids)
        {
            for (size_t i = 0; i < dominant_normals_.size (); ++i)
                centroids.push_back (dominant_normals_[i]);
        }

        /** \brief Sets max. Euclidean distance between points to be added to the cluster
         * \param[in] d the maximum Euclidean distance
         */

        inline void
        setClusterTolerance (float d)
        {
            cluster_tolerance_ = d;
        }

        /** \brief Sets max. deviation of the normals between two points so they can be clustered together
         * \param[in] d the maximum deviation
         */
        inline void
        setEPSAngleThreshold (float d)
        {
            eps_angle_threshold_ = d;
        }

        /** \brief Sets curvature threshold for removing normals
         * \param[in] d the curvature threshold
         */
        inline void
        setCurvatureThreshold (float d)
        {
            curv_threshold_ = d;
        }

        /** \brief Set minimum amount of points for a cluster to be considered
         * \param[in] min the minimum amount of points to be set
         */
        inline void
        setMinPoints (size_t min)
        {
            min_points_ = min;
        }

        /** \brief Sets wether if the signatures should be normalized or not
         * \param[in] normalize true if normalization is required, false otherwise
         */
        inline void
        setNormalizeBins (bool normalize)
        {
            normalize_bins_ = normalize;
        }

        /** \brief Gets the indices of the original point cloud used to compute the signatures
         * \param[out] indices vector of point indices
         */
        inline void
        getClusterIndices (std::vector<pcl::PointIndices> & indices)
        {
            indices = clusters_;
        }

        /** \brief Gets the number of non-disambiguable axes that correspond to each centroid
         * \param[out] cluster_axes vector mapping each centroid to the number of signatures
         */
        inline void
        getClusterAxes (std::vector<short> & cluster_axes)
        {
            cluster_axes = cluster_axes_;
        }

        /** \brief Sets the refinement factor for the clusters
         * \param[in] rc the factor used to decide if a point is used to estimate a stable cluster
         */
        void
        setRefineClusters (float rc)
        {
            refine_clusters_ = rc;
        }

        /** \brief Returns the transformations aligning the point cloud to the corresponding SGURF
         * \param[out] trans vector of transformations
         */
        void
        getTransforms (std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & trans)
        {
            trans = transforms_;
        }

        /** \brief Returns a boolean vector indicating of the transformation obtained by getTransforms() represents
         * a valid SGURF
         * \param[out] valid vector of booleans
         */
        void
        getValidTransformsVec (std::vector<bool> & valid)
        {
            valid = valid_transforms_;
        }

        /** \brief Sets the min axis ratio between the SGURF axes to decide if disambiguition is feasible
         * \param[in] f the ratio between axes
         */
        void
        setAxisRatio (float f)
        {
            axis_ratio_ = f;
        }

        /** \brief Sets the min disambiguition axis value to generate several SGURFs for the cluster when disambiguition is difficult
         * \param[in] f the min axis value
         */
        void
        setMinAxisValue (float f)
        {
            min_axis_value_ = f;
        }

        /** \brief Overloaded computed method from pcl::Feature.
         * \param[out] output the resultant point cloud model dataset containing the estimated features
         */
        void
        compute (PointCloudOut &output);

    private:
        /** \brief Values describing the viewpoint ("pinhole" camera model assumed).
         * By default, the viewpoint is set to 0,0,0.
         */
        float vpx_, vpy_, vpz_;

        /** \brief Size of the voxels after voxel gridding. IMPORTANT: Must match the voxel
         * size of the training data or the normalize_bins_ flag must be set to true.
         */
        float leaf_size_;

        /** \brief Wether to normalize the signatures or not. Default: false. */
        bool normalize_bins_;

        /** \brief Curvature threshold for removing normals. */
        float curv_threshold_;

        /** \brief allowed Euclidean distance between points to be added to the cluster. */
        float cluster_tolerance_;

        /** \brief deviation of the normals between two points so they can be clustered together. */
        float eps_angle_threshold_;

        /** \brief Minimum amount of points in a clustered region to be considered stable for CVFH
         * computation.
         */
        size_t min_points_;

        /** \brief Radius for the normals computation. */
        float radius_normals_;

        /** \brief Factor for the cluster refinement */
        float refine_clusters_;

        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_;
        std::vector<bool> valid_transforms_;

        float axis_ratio_;
        float min_axis_value_;

        /** \brief Estimate the OUR-CVFH descriptors at
         * a set of points given by <setInputCloud (), setIndices ()> using the surface in
         * setSearchSurface ()
         *
         * \param[out] output the resultant point cloud model dataset that contains the OUR-CVFH
         * feature estimates
         */
        void
        computeFeature (PointCloudOut &output);

        /** \brief Region growing method using Euclidean distances and neighbors normals to
         * add points to a region.
         * \param[in] cloud point cloud to split into regions
         * \param[in] normals are the normals of cloud
         * \param[in] tolerance is the allowed Euclidean distance between points to be added to
         * the cluster
         * \param[in] tree is the spatial search structure for nearest neighbour search
         * \param[out] clusters vector of indices representing the clustered regions
         * \param[in] eps_angle deviation of the normals between two points so they can be
         * clustered together
         * \param[in] min_pts_per_cluster minimum cluster size. (default: 1 point)
         * \param[in] max_pts_per_cluster maximum cluster size. (default: all the points)
         */
        void
        extractEuclideanClustersSmooth (const pcl::PointCloud<pcl::PointNormal> &cloud, const pcl::PointCloud<pcl::PointNormal> &normals,
                                        float tolerance, const pcl::search::Search<pcl::PointNormal>::Ptr &tree,
                                        std::vector<pcl::PointIndices> &clusters, double eps_angle, unsigned int min_pts_per_cluster = 1,
                                        unsigned int max_pts_per_cluster = (std::numeric_limits<int>::max) ());

    protected:
        /** \brief Centroids that were used to compute different OUR-CVFH descriptors */
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > centroids_dominant_orientations_;
        /** \brief Normal centroids that were used to compute different OUR-CVFH descriptors */
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > dominant_normals_;
        /** \brief Indices to the points representing the stable clusters */
        std::vector<pcl::PointIndices> clusters_;
        /** \brief Mapping from clusters to OUR-CVFH descriptors */
        std::vector<short> cluster_axes_;
    };
}

/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id: cvfh.hpp 5311 2012-03-26 22:02:04Z aaldoma $
 *
 */

#include <pcl/features/our_cvfh.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh_tools.h>
#include <pcl/common/transforms.h>

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointNT, typename PointOutT> void
pcl_fixed::OURCVFHEstimation2<PointInT, PointNT, PointOutT>::compute (PointCloudOut &output)
{
    if (!Feature<PointInT, PointOutT>::initCompute ())
    {
        output.width = output.height = 0;
        output.points.clear ();
        return;
    }
    // Resize the output dataset
    // Important! We should only allocate precisely how many elements we will need, otherwise
    // we risk at pre-allocating too much memory which could lead to bad_alloc
    // (see http://dev.pointclouds.org/issues/657)
    output.width = output.height = 1;
    output.points.resize (1);

    // Perform the actual feature computation
    computeFeature (output);

    Feature<PointInT, PointOutT>::deinitCompute ();
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointNT, typename PointOutT> void
pcl_fixed::OURCVFHEstimation2<PointInT, PointNT, PointOutT>::extractEuclideanClustersSmooth (const pcl::PointCloud<pcl::PointNormal> &cloud,
                                                                                      const pcl::PointCloud<pcl::PointNormal> &normals,
                                                                                      float tolerance,
                                                                                      const pcl::search::Search<pcl::PointNormal>::Ptr &tree,
                                                                                      std::vector<pcl::PointIndices> &clusters, double eps_angle,
                                                                                      unsigned int min_pts_per_cluster,
                                                                                      unsigned int max_pts_per_cluster)
{
    if (tree->getInputCloud ()->points.size () != cloud.points.size ())
    {
        PCL_ERROR ("[pcl::extractEuclideanClusters] Tree built for a different point cloud dataset (%lu) than the input cloud (%lu)!\n", tree->getInputCloud ()->points.size (), cloud.points.size ());
        return;
    }
    if (cloud.points.size () != normals.points.size ())
    {
        PCL_ERROR ("[pcl::extractEuclideanClusters] Number of points in the input point cloud (%lu) different than normals (%lu)!\n", cloud.points.size (), normals.points.size ());
        return;
    }

    // Create a bool vector of processed point indices, and initialize it to false
    std::vector<bool> processed (cloud.points.size (), false);

    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    // Process all points in the indices vector
    for (int i = 0; i < static_cast<int> (cloud.points.size ()); ++i)
    {
        if (processed[i])
            continue;

        std::vector<unsigned int> seed_queue;
        int sq_idx = 0;
        seed_queue.push_back (i);

        processed[i] = true;

        while (sq_idx < static_cast<int> (seed_queue.size ()))
        {
            // Search for sq_idx
            if (!tree->radiusSearch (seed_queue[sq_idx], tolerance, nn_indices, nn_distances))
            {
                sq_idx++;
                continue;
            }

            for (size_t j = 1; j < nn_indices.size (); ++j) // nn_indices[0] should be sq_idx
            {
                if (processed[nn_indices[j]]) // Has this point been processed before ?
                    continue;

                //processed[nn_indices[j]] = true;
                // [-1;1]

                double dot_p = normals.points[seed_queue[sq_idx]].normal[0] * normals.points[nn_indices[j]].normal[0]
                               + normals.points[seed_queue[sq_idx]].normal[1] * normals.points[nn_indices[j]].normal[1] + normals.points[seed_queue[sq_idx]].normal[2]
                                                                                                                          * normals.points[nn_indices[j]].normal[2];

                if (fabs (acos (dot_p)) < eps_angle)
                {
                    processed[nn_indices[j]] = true;
                    seed_queue.push_back (nn_indices[j]);
                }
            }

            sq_idx++;
        }

        // If this queue is satisfactory, add to the clusters
        if (seed_queue.size () >= min_pts_per_cluster && seed_queue.size () <= max_pts_per_cluster)
        {
            pcl::PointIndices r;
            r.indices.resize (seed_queue.size ());
            for (size_t j = 0; j < seed_queue.size (); ++j)
                r.indices[j] = seed_queue[j];

            std::sort (r.indices.begin (), r.indices.end ());
            r.indices.erase (std::unique (r.indices.begin (), r.indices.end ()), r.indices.end ());

            r.header = cloud.header;
            clusters.push_back (r); // We could avoid a copy by working directly in the vector
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointNT, typename PointOutT> void
pcl_fixed::OURCVFHEstimation2<PointInT, PointNT, PointOutT>::filterNormalsWithHighCurvature (const pcl::PointCloud<PointNT> & cloud,
                                                                                      std::vector<int> &indices_to_use,
                                                                                      std::vector<int> &indices_out, std::vector<int> &indices_in,
                                                                                      float threshold)
{
    indices_out.resize (cloud.points.size ());
    indices_in.resize (cloud.points.size ());

    size_t in, out;
    in = out = 0;

    for (int i = 0; i < static_cast<int> (indices_to_use.size ()); i++)
    {
        if (cloud.points[indices_to_use[i]].curvature > threshold)
        {
            indices_out[out] = indices_to_use[i];
            out++;
        }
        else
        {
            indices_in[in] = indices_to_use[i];
            in++;
        }
    }

    indices_out.resize (out);
    indices_in.resize (in);
}

template<typename PointInT, typename PointNT, typename PointOutT> bool
pcl_fixed::OURCVFHEstimation2<PointInT, PointNT, PointOutT>::sgurf (Eigen::Vector3f & centroid, Eigen::Vector3f & normal_centroid,
                                                             PointInTPtr & processed, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > & transformations,
                                                             PointInTPtr & grid, pcl::PointIndices & indices)
{

    Eigen::Vector3f plane_normal;
    plane_normal[0] = -centroid[0];
    plane_normal[1] = -centroid[1];
    plane_normal[2] = -centroid[2];
    Eigen::Vector3f z_vector = Eigen::Vector3f::UnitZ ();
    plane_normal.normalize ();
    Eigen::Vector3f axis = plane_normal.cross (z_vector);
    double rotation = -asin (axis.norm ());
    axis.normalize ();

    Eigen::Affine3f transformPC (Eigen::AngleAxisf (static_cast<float> (rotation), axis));

    grid->points.resize (processed->points.size ());
    for (size_t k = 0; k < processed->points.size (); k++)
        grid->points[k].getVector4fMap () = processed->points[k].getVector4fMap ();

    pcl::transformPointCloud (*grid, *grid, transformPC);

    Eigen::Vector4f centroid4f (centroid[0], centroid[1], centroid[2], 0);
    Eigen::Vector4f normal_centroid4f (normal_centroid[0], normal_centroid[1], normal_centroid[2], 0);

    centroid4f = transformPC * centroid4f;
    normal_centroid4f = transformPC * normal_centroid4f;

    Eigen::Vector3f centroid3f (centroid4f[0], centroid4f[1], centroid4f[2]);

    Eigen::Vector4f farthest_away;
    pcl::getMaxDistance (*grid, indices.indices, centroid4f, farthest_away);
    farthest_away[3] = 0;

    float max_dist = (farthest_away - centroid4f).norm ();

    pcl::demeanPointCloud (*grid, centroid4f, *grid);

    Eigen::Matrix4f center_mat;
    center_mat.setIdentity (4, 4);
    center_mat (0, 3) = -centroid4f[0];
    center_mat (1, 3) = -centroid4f[1];
    center_mat (2, 3) = -centroid4f[2];

    Eigen::Matrix3f scatter;
    scatter.setZero ();
    float sum_w = 0.f;

    //for (int k = 0; k < static_cast<intgrid->points[k].getVector3fMap ();> (grid->points.size ()); k++)
    for (int k = 0; k < static_cast<int> (indices.indices.size ()); k++)
    {
        Eigen::Vector3f pvector = grid->points[indices.indices[k]].getVector3fMap ();
        float d_k = (pvector).norm ();
        float w = (max_dist - d_k);
        Eigen::Vector3f diff = (pvector);
        Eigen::Matrix3f mat = diff * diff.transpose ();
        scatter = scatter + mat * w;
        sum_w += w;
    }

    scatter /= sum_w;

    Eigen::JacobiSVD <Eigen::MatrixXf> svd (scatter, Eigen::ComputeFullV);
    Eigen::Vector3f evx = svd.matrixV ().col (0);
    Eigen::Vector3f evy = svd.matrixV ().col (1);
    Eigen::Vector3f evz = svd.matrixV ().col (2);
    Eigen::Vector3f evxminus = evx * -1;
    Eigen::Vector3f evyminus = evy * -1;
    Eigen::Vector3f evzminus = evz * -1;

    float s_xplus, s_xminus, s_yplus, s_yminus;
    s_xplus = s_xminus = s_yplus = s_yminus = 0.f;

    //disambiguate rf using all points
    for (int k = 0; k < static_cast<int> (grid->points.size ()); k++)
    {
        Eigen::Vector3f pvector = grid->points[k].getVector3fMap ();
        float dist_x, dist_y;
        dist_x = std::abs (evx.dot (pvector));
        dist_y = std::abs (evy.dot (pvector));

        if ((pvector).dot (evx) >= 0)
            s_xplus += dist_x;
        else
            s_xminus += dist_x;

        if ((pvector).dot (evy) >= 0)
            s_yplus += dist_y;
        else
            s_yminus += dist_y;

    }

    if (s_xplus < s_xminus)
        evx = evxminus;

    if (s_yplus < s_yminus)
        evy = evyminus;

    //select the axis that could be disambiguated more easily
    float fx, fy;
    float max_x = static_cast<float> (std::max (s_xplus, s_xminus));
    float min_x = static_cast<float> (std::min (s_xplus, s_xminus));
    float max_y = static_cast<float> (std::max (s_yplus, s_yminus));
    float min_y = static_cast<float> (std::min (s_yplus, s_yminus));

    fx = (min_x / max_x);
    fy = (min_y / max_y);

    Eigen::Vector3f normal3f = Eigen::Vector3f (normal_centroid4f[0], normal_centroid4f[1], normal_centroid4f[2]);
    if (normal3f.dot (evz) < 0)
        evz = evzminus;

    //if fx/y close to 1, it was hard to disambiguate
    //what if both are equally easy or difficult to disambiguate, namely fy == fx or very close

    float max_axis = std::max (fx, fy);
    float min_axis = std::min (fx, fy);

    if ((min_axis / max_axis) > axis_ratio_)
    {
        PCL_WARN ("Both axes are equally easy/difficult to disambiguate\n");

        Eigen::Vector3f evy_copy = evy;
        Eigen::Vector3f evxminus = evx * -1;
        Eigen::Vector3f evyminus = evy * -1;

        if (min_axis > min_axis_value_)
        {
            //combination of all possibilities
            evy = evx.cross (evz);
            Eigen::Matrix4f trans = createTransFromAxes (evx, evy, evz, transformPC, center_mat);
            transformations.push_back (trans);

            evx = evxminus;
            evy = evx.cross (evz);
            trans = createTransFromAxes (evx, evy, evz, transformPC, center_mat);
            transformations.push_back (trans);

            evx = evy_copy;
            evy = evx.cross (evz);
            trans = createTransFromAxes (evx, evy, evz, transformPC, center_mat);
            transformations.push_back (trans);

            evx = evyminus;
            evy = evx.cross (evz);
            trans = createTransFromAxes (evx, evy, evz, transformPC, center_mat);
            transformations.push_back (trans);

        }
        else
        {
            //1-st case (evx selected)
            evy = evx.cross (evz);
            Eigen::Matrix4f trans = createTransFromAxes (evx, evy, evz, transformPC, center_mat);
            transformations.push_back (trans);

            //2-nd case (evy selected)
            evx = evy_copy;
            evy = evx.cross (evz);
            trans = createTransFromAxes (evx, evy, evz, transformPC, center_mat);
            transformations.push_back (trans);
        }
    }
    else
    {
        if (fy < fx)
        {
            evx = evy;
            fx = fy;
        }

        evy = evx.cross (evz);
        Eigen::Matrix4f trans = createTransFromAxes (evx, evy, evz, transformPC, center_mat);
        transformations.push_back (trans);

    }

    return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointNT, typename PointOutT> void
pcl_fixed::OURCVFHEstimation2<PointInT, PointNT, PointOutT>::computeRFAndShapeDistribution (PointInTPtr & processed, PointCloudOut & output,
                                                                                     std::vector<pcl::PointIndices> & cluster_indices)
{
    PointCloudOut ourcvfh_output;

    cluster_axes_.clear ();
    cluster_axes_.resize (centroids_dominant_orientations_.size ());

    for (size_t i = 0; i < centroids_dominant_orientations_.size (); i++)
    {

        std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations;
        PointInTPtr grid (new pcl::PointCloud<PointInT>);
        sgurf (centroids_dominant_orientations_[i], dominant_normals_[i], processed, transformations, grid, cluster_indices[i]);

        // Make a note of how many transformations correspond to each cluster
        cluster_axes_[i] = transformations.size ();

        for (size_t t = 0; t < transformations.size (); t++)
        {

            pcl::transformPointCloud (*processed, *grid, transformations[t]);
            transforms_.push_back (transformations[t]);
            valid_transforms_.push_back (true);

            std::vector < Eigen::VectorXf > quadrants (8);
            int size_hists = 13;
            int num_hists = 8;
            for (int k = 0; k < num_hists; k++)
                quadrants[k].setZero (size_hists);

            Eigen::Vector4f centroid_p;
            centroid_p.setZero ();
            Eigen::Vector4f max_pt;
            pcl::getMaxDistance (*grid, centroid_p, max_pt);
            max_pt[3] = 0;
            double distance_normalization_factor = (centroid_p - max_pt).norm ();

            float hist_incr;
            if (normalize_bins_)
                hist_incr = 100.0f / static_cast<float> (grid->points.size () - 1);
            else
                hist_incr = 1.0f;

            float * weights = new float[num_hists];
            float sigma = 0.01f; //1cm
            float sigma_sq = sigma * sigma;

            for (int k = 0; k < static_cast<int> (grid->points.size ()); k++)
            {
                Eigen::Vector4f p = grid->points[k].getVector4fMap ();
                p[3] = 0.f;
                float d = p.norm ();

                //compute weight for all octants
                float wx = 1.f - std::exp (-((p[0] * p[0]) / (2.f * sigma_sq))); //how is the weight distributed among two semi-cubes
                float wy = 1.f - std::exp (-((p[1] * p[1]) / (2.f * sigma_sq)));
                float wz = 1.f - std::exp (-((p[2] * p[2]) / (2.f * sigma_sq)));

                //distribute the weights using the x-coordinate
                if (p[0] >= 0)
                {
                    for (size_t ii = 0; ii <= 3; ii++)
                        weights[ii] = 0.5f - wx * 0.5f;

                    for (size_t ii = 4; ii <= 7; ii++)
                        weights[ii] = 0.5f + wx * 0.5f;
                }
                else
                {
                    for (size_t ii = 0; ii <= 3; ii++)
                        weights[ii] = 0.5f + wx * 0.5f;

                    for (size_t ii = 4; ii <= 7; ii++)
                        weights[ii] = 0.5f - wx * 0.5f;
                }

                //distribute the weights using the y-coordinate
                if (p[1] >= 0)
                {
                    for (size_t ii = 0; ii <= 1; ii++)
                        weights[ii] *= 0.5f - wy * 0.5f;
                    for (size_t ii = 4; ii <= 5; ii++)
                        weights[ii] *= 0.5f - wy * 0.5f;

                    for (size_t ii = 2; ii <= 3; ii++)
                        weights[ii] *= 0.5f + wy * 0.5f;

                    for (size_t ii = 6; ii <= 7; ii++)
                        weights[ii] *= 0.5f + wy * 0.5f;
                }
                else
                {
                    for (size_t ii = 0; ii <= 1; ii++)
                        weights[ii] *= 0.5f + wy * 0.5f;
                    for (size_t ii = 4; ii <= 5; ii++)
                        weights[ii] *= 0.5f + wy * 0.5f;

                    for (size_t ii = 2; ii <= 3; ii++)
                        weights[ii] *= 0.5f - wy * 0.5f;

                    for (size_t ii = 6; ii <= 7; ii++)
                        weights[ii] *= 0.5f - wy * 0.5f;
                }

                //distribute the weights using the z-coordinate
                if (p[2] >= 0)
                {
                    for (size_t ii = 0; ii <= 7; ii += 2)
                        weights[ii] *= 0.5f - wz * 0.5f;

                    for (size_t ii = 1; ii <= 7; ii += 2)
                        weights[ii] *= 0.5f + wz * 0.5f;

                }
                else
                {
                    for (size_t ii = 0; ii <= 7; ii += 2)
                        weights[ii] *= 0.5f + wz * 0.5f;

                    for (size_t ii = 1; ii <= 7; ii += 2)
                        weights[ii] *= 0.5f - wz * 0.5f;
                }

                int h_index = (d <= 0) ? 0 : std::ceil (size_hists * (d / distance_normalization_factor)) - 1;
                /* from http://www.pcl-users.org/OUR-CVFH-problem-td4028436.html
                   h_index will be 13 when d is computed on the farthest away point.

                  adding the following after computing h_index fixes the problem:
                */
                if(h_index > 12)
                    h_index = 12;
                for (int j = 0; j < num_hists; j++)
                    quadrants[j][h_index] += hist_incr * weights[j];

            }

            //copy to the cvfh signature
            PointCloudOut vfh_signature;
            vfh_signature.points.resize (1);
            vfh_signature.width = vfh_signature.height = 1;
            for (int d = 0; d < 308; ++d)
                vfh_signature.points[0].histogram[d] = output.points[i].histogram[d];

            int pos = 45 * 3;
            for (int k = 0; k < num_hists; k++)
            {
                for (int ii = 0; ii < size_hists; ii++, pos++)
                {
                    vfh_signature.points[0].histogram[pos] = quadrants[k][ii];
                }
            }

            ourcvfh_output.points.push_back (vfh_signature.points[0]);
            ourcvfh_output.width = ourcvfh_output.points.size ();
            delete[] weights;
        }
    }

    if (ourcvfh_output.points.size ())
    {
        ourcvfh_output.height = 1;
    }
    output = ourcvfh_output;
}

//////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointInT, typename PointNT, typename PointOutT> void
pcl_fixed::OURCVFHEstimation2<PointInT, PointNT, PointOutT>::computeFeature (PointCloudOut &output)
{
    if (refine_clusters_ <= 0.f)
        refine_clusters_ = 1.f;

    // Check if input was set
    if (!normals_)
    {
        PCL_ERROR ("[pcl::%s::computeFeature] No input dataset containing normals was given!\n", getClassName ().c_str ());
        output.width = output.height = 0;
        output.points.clear ();
        return;
    }
    if (normals_->points.size () != surface_->points.size ())
    {
        PCL_ERROR ("[pcl::%s::computeFeature] The number of points in the input dataset differs from the number of points in the dataset containing the normals!\n", getClassName ().c_str ());
        output.width = output.height = 0;
        output.points.clear ();
        return;
    }

    centroids_dominant_orientations_.clear ();
    clusters_.clear ();
    transforms_.clear ();
    dominant_normals_.clear ();

    // ---[ Step 0: remove normals with high curvature
    std::vector<int> indices_out;
    std::vector<int> indices_in;
    filterNormalsWithHighCurvature (*normals_, *indices_, indices_out, indices_in, curv_threshold_);

    pcl::PointCloud<pcl::PointNormal>::Ptr normals_filtered_cloud (new pcl::PointCloud<pcl::PointNormal> ());
    normals_filtered_cloud->width = static_cast<uint32_t> (indices_in.size ());
    normals_filtered_cloud->height = 1;
    normals_filtered_cloud->points.resize (normals_filtered_cloud->width);

    std::vector<int> indices_from_nfc_to_indices;
    indices_from_nfc_to_indices.resize (indices_in.size ());

    for (size_t i = 0; i < indices_in.size (); ++i)
    {
        normals_filtered_cloud->points[i].x = surface_->points[indices_in[i]].x;
        normals_filtered_cloud->points[i].y = surface_->points[indices_in[i]].y;
        normals_filtered_cloud->points[i].z = surface_->points[indices_in[i]].z;
        //normals_filtered_cloud->points[i].getNormalVector4fMap() = normals_->points[indices_in[i]].getNormalVector4fMap();
        indices_from_nfc_to_indices[i] = indices_in[i];
    }

    std::vector<pcl::PointIndices> clusters;

    if (normals_filtered_cloud->points.size () >= min_points_)
    {
        //recompute normals and use them for clustering
        {
            KdTreePtr normals_tree_filtered (new pcl::search::KdTree<pcl::PointNormal> (false));
            normals_tree_filtered->setInputCloud (normals_filtered_cloud);
            pcl::NormalEstimation<PointNormal, PointNormal> n3d;
            n3d.setRadiusSearch (radius_normals_);
            n3d.setSearchMethod (normals_tree_filtered);
            n3d.setInputCloud (normals_filtered_cloud);
            n3d.compute (*normals_filtered_cloud);
        }

        KdTreePtr normals_tree (new pcl::search::KdTree<pcl::PointNormal> (false));
        normals_tree->setInputCloud (normals_filtered_cloud);

        extractEuclideanClustersSmooth (*normals_filtered_cloud, *normals_filtered_cloud, cluster_tolerance_, normals_tree, clusters,
                                        eps_angle_threshold_, static_cast<unsigned int> (min_points_));

        std::vector<pcl::PointIndices> clusters_filtered;
        int cluster_filtered_idx = 0;
        for (size_t i = 0; i < clusters.size (); i++)
        {

            pcl::PointIndices pi;
            pcl::PointIndices pi_cvfh;
            pcl::PointIndices pi_filtered;

            clusters_.push_back (pi);
            clusters_filtered.push_back (pi_filtered);

            Eigen::Vector4f avg_normal = Eigen::Vector4f::Zero ();
            Eigen::Vector4f avg_centroid = Eigen::Vector4f::Zero ();

            for (size_t j = 0; j < clusters[i].indices.size (); j++)
            {
                avg_normal += normals_filtered_cloud->points[clusters[i].indices[j]].getNormalVector4fMap ();
                avg_centroid += normals_filtered_cloud->points[clusters[i].indices[j]].getVector4fMap ();
            }

            avg_normal /= static_cast<float> (clusters[i].indices.size ());
            avg_centroid /= static_cast<float> (clusters[i].indices.size ());
            avg_normal.normalize ();

            Eigen::Vector3f avg_norm (avg_normal[0], avg_normal[1], avg_normal[2]);
            Eigen::Vector3f avg_dominant_centroid (avg_centroid[0], avg_centroid[1], avg_centroid[2]);

            for (size_t j = 0; j < clusters[i].indices.size (); j++)
            {
                //decide if normal should be added
                double dot_p = avg_normal.dot (normals_filtered_cloud->points[clusters[i].indices[j]].getNormalVector4fMap ());
                if (fabs (acos (dot_p)) < (eps_angle_threshold_ * refine_clusters_))
                {
                    clusters_[cluster_filtered_idx].indices.push_back (indices_from_nfc_to_indices[clusters[i].indices[j]]);
                    clusters_filtered[cluster_filtered_idx].indices.push_back (clusters[i].indices[j]);
                }
            }

            //remove last cluster if no points found...
            if (clusters_[cluster_filtered_idx].indices.size () == 0)
            {
                clusters_.pop_back ();
                clusters_filtered.pop_back ();
            }
            else
                cluster_filtered_idx++;
        }

        clusters = clusters_filtered;

    }

    pcl::VFHEstimation<PointInT, PointNT, pcl::VFHSignature308> vfh;
    vfh.setInputCloud (surface_);
    vfh.setInputNormals (normals_);
    vfh.setIndices (indices_);
    vfh.setSearchMethod (this->tree_);
    vfh.setUseGivenNormal (true);
    vfh.setUseGivenCentroid (true);
    vfh.setNormalizeBins (normalize_bins_);
    output.height = 1;

    // ---[ Step 1b : check if any dominant cluster was found
    if (clusters.size () > 0)
    { // ---[ Step 1b.1 : If yes, compute CVFH using the cluster information

        for (size_t i = 0; i < clusters.size (); ++i) //for each cluster

        {
            Eigen::Vector4f avg_normal = Eigen::Vector4f::Zero ();
            Eigen::Vector4f avg_centroid = Eigen::Vector4f::Zero ();

            for (size_t j = 0; j < clusters[i].indices.size (); j++)
            {
                avg_normal += normals_filtered_cloud->points[clusters[i].indices[j]].getNormalVector4fMap ();
                avg_centroid += normals_filtered_cloud->points[clusters[i].indices[j]].getVector4fMap ();
            }

            avg_normal /= static_cast<float> (clusters[i].indices.size ());
            avg_centroid /= static_cast<float> (clusters[i].indices.size ());
            avg_normal.normalize ();

            Eigen::Vector3f avg_norm (avg_normal[0], avg_normal[1], avg_normal[2]);
            Eigen::Vector3f avg_dominant_centroid (avg_centroid[0], avg_centroid[1], avg_centroid[2]);

            //append normal and centroid for the clusters
            dominant_normals_.push_back (avg_norm);
            centroids_dominant_orientations_.push_back (avg_dominant_centroid);
        }

        //compute modified VFH for all dominant clusters and add them to the list!
        output.points.resize (dominant_normals_.size ());
        output.width = static_cast<uint32_t> (dominant_normals_.size ());

        for (size_t i = 0; i < dominant_normals_.size (); ++i)
        {
            //configure VFH computation for CVFH
            vfh.setNormalToUse (dominant_normals_[i]);
            vfh.setCentroidToUse (centroids_dominant_orientations_[i]);
            pcl::PointCloud<pcl::VFHSignature308> vfh_signature;
            vfh.compute (vfh_signature);
            output.points[i] = vfh_signature.points[0];
        }

        //finish filling the descriptor with the shape distribution
        PointInTPtr cloud_input (new pcl::PointCloud<PointInT>);
        pcl::copyPointCloud (*surface_, *indices_, *cloud_input);
        computeRFAndShapeDistribution (cloud_input, output, clusters_); //this will set transforms_
    }
    else
    { // ---[ Step 1b.1 : If no, compute a VFH using all the object points

        PCL_WARN("No clusters were found in the surface... using VFH...\n");
        Eigen::Vector4f avg_centroid;
        pcl::compute3DCentroid (*surface_, avg_centroid);
        Eigen::Vector3f cloud_centroid (avg_centroid[0], avg_centroid[1], avg_centroid[2]);
        centroids_dominant_orientations_.push_back (cloud_centroid);

        //configure VFH computation using all object points
        vfh.setCentroidToUse (cloud_centroid);
        vfh.setUseGivenNormal (false);

        pcl::PointCloud<pcl::VFHSignature308> vfh_signature;
        vfh.compute (vfh_signature);

        output.points.resize (1);
        output.width = 1;

        output.points[0] = vfh_signature.points[0];
        Eigen::Matrix4f id = Eigen::Matrix4f::Identity ();
        transforms_.push_back (id);
        valid_transforms_.push_back (false);
    }
}
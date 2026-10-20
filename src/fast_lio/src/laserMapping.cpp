// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <chrono>
#include <sys/stat.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#ifdef FASTLIO_REPLAY
#include "replay.hpp"
#endif
#include <ikd-Tree/ikd_Tree.h>
#include <voxel_map.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)

/*** Timing log (runtime_pos_log_enable): one row per processed scan, Log/fast_lio_time_log.csv at exit ***/
struct TimeLogRow
{
    double stamp, total, incremental, search, delete_time, preprocess;
    int scan_points, delete_size, tree_size_st, tree_size_end, add_points, down_points;
    int matched[5];             // last update iteration: points per PointStatus
    int update_calls;           // h_share_model calls (iterations) of the scan
    double residual_mean;       // mean |point-to-plane residual| of the matched points [m]
    int map_new, map_rejected, map_candidates;
    int voxel_matched;          // of the matched points: planes from the voxel map
};
vector<TimeLogRow> time_log;
string time_log_path;                     // runtime_pos_log_path; empty: <package source>/Log/fast_lio_time_log.csv
deque<double> preprocess_time_buffer;     // preprocessing time of each buffered scan, mirrors lidar_buffer
double current_preprocess_time = 0.0;     // of the scan being processed
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double match_time = 0, solve_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0, filter_size_map_min = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0;
int    omp_threads = MP_PROC_NUM;   // num_threads: OpenMP threads of the matching loop
bool   lidar_pushed, flg_first_scan = true, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;

vector<BoxPointType> cub_needrm;
// Per point of the downsampled scan, sized in the main loop. point_selected_surf is written by the
// OpenMP loop in h_share_model, so it must not be a vector<bool> (bit-packed: concurrent writes race).
vector<PointVector>  Nearest_Points;
vector<uint8_t>      point_selected_surf;
vector<float>        res_last;
// Plane fitted to each point's neighbours, and whether the fit passed. Fitted when the neighbours are
// searched, reused by the update iterations that reuse them (the plane depends on the neighbours only).
vector<VF(4), Eigen::aligned_allocator<VF(4)>> plane_coef;
vector<uint8_t>      plane_ok;
// Diagnostics (timing log): why each downsampled point does or does not give a residual, and what
// map_incremental does with the points.
enum PointStatus : uint8_t { PT_MATCHED, PT_NO_NEIGHBOURS, PT_FAR_NEIGHBOURS, PT_PLANE_FAIL, PT_RESIDUAL_FAIL };
vector<uint8_t>      point_status;
int match_counts[5] = {0}, update_calls = 0;
int map_new = 0, map_rejected = 0, map_candidates = 0;
// mapping.map_insert_rule: "closest_to_centre" (upstream: a point closer to the voxel centre replaces the
// stored one) or "first_point" (the first point in a voxel stays; no grid-dependent bias, see
// results/mac/map_analysis).
bool map_keep_first_point = false;
// map_type: "ikdtree" (upstream: points, 5-NN plane fit per query), "voxel" (plane statistics per voxel,
// see voxel_map.h) or "hybrid" (both: a trustworthy voxel plane where there is one, otherwise the 5-NN fit).
bool use_ikdtree = true, use_voxel_planes = false;
VoxelMap voxel_map;
vector<uint8_t> plane_from_voxel;   // per point: its plane came from the voxel map (no 5-NN fit needed)
int voxel_matched = 0;              // points matched to a voxel plane (last update iteration)
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
// Bounded input queues: if processing falls behind or one sensor stops, the oldest data goes first.
const size_t MAX_LIDAR_BUFFER = 100;    // 10 s of scans at 10 Hz
const size_t MAX_IMU_BUFFER   = 2000;   // 10 s at 200 Hz

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI());
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;

KD_TREE<PointType> ikdtree;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
uint64_t odom_count = 0;                  // odometry messages published so far
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    // Not dead code: drains ikd-tree's list of deleted points, which otherwise grows forever.
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

// True at most once per 5 s for one call site (`last` is that call site's static).
bool warn_now(chrono::steady_clock::time_point &last)
{
    const auto now = chrono::steady_clock::now();
    if (now - last < chrono::seconds(5))
        return false;
    last = now;
    return true;
}

// Drops the oldest queued scans until `keep` are left. The scan, stamp and preprocessing-time queues
// hold one entry per scan and stay paired; a scan that sync_packages had started on is re-read.
void drop_queued_scans(size_t keep)
{
    if (lidar_buffer.size() <= keep)
        return;
    while (lidar_buffer.size() > keep)
    {
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        if (!preprocess_time_buffer.empty())
            preprocess_time_buffer.pop_front();
    }
    lidar_pushed = false;
}

// Queues one preprocessed scan (caller holds mtx_buffer).
void enqueue_scan(const PointCloudXYZI::Ptr &scan, double stamp, double preprocess_time)
{
    lidar_buffer.push_back(scan);
    time_buffer.push_back(stamp);
    preprocess_time_buffer.push_back(preprocess_time);
    if (lidar_buffer.size() > MAX_LIDAR_BUFFER)
    {
        static chrono::steady_clock::time_point warned;
        if (warn_now(warned))
            std::cerr << "Lidar queue full (processing too slow?): dropping the oldest scans" << std::endl;
        drop_queued_scans(MAX_LIDAR_BUFFER);
    }
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        drop_queued_scans(0);
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    last_timestamp_lidar = cur_time;
    enqueue_scan(ptr, cur_time, omp_get_wtime() - preprocess_start_time);
    mtx_buffer.unlock();
}

void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        drop_queued_scans(0);
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;

    if (abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    enqueue_scan(ptr, last_timestamp_lidar, omp_get_wtime() - preprocess_start_time);
    mtx_buffer.unlock();
}

void imu_cbk(sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    sensor_msgs::msg::Imu::SharedPtr msg(std::move(msg_in));
    // Lidar-to-IMU time offset, applied in integer nanoseconds: the stamp is kept exactly (it used to
    // go through double seconds and back, moving 94 % of stamps by up to 240 ns).
    if (time_diff_lidar_to_imu != 0.0)
        msg->header.stamp = rclcpp::Time(msg->header.stamp) -
                            rclcpp::Duration(std::chrono::nanoseconds(llround(time_diff_lidar_to_imu * 1e9)));

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "IMU loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    if (imu_buffer.size() > MAX_IMU_BUFFER)
    {
        static chrono::steady_clock::time_point warned;
        if (warn_now(warned))
            std::cerr << "IMU queue full (no lidar scans?): dropping the oldest samples" << std::endl;
        imu_buffer.pop_front();
    }
    mtx_buffer.unlock();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    if (!preprocess_time_buffer.empty())
    {
        current_preprocess_time = preprocess_time_buffer.front();
        preprocess_time_buffer.pop_front();
    }
    lidar_pushed = false;
    return true;
}

// Voxel map: every kept, deskewed point of the scan goes into the statistics of its voxel.
void voxel_map_incremental()
{
    const PointCloudXYZI &scan = *feats_undistort;
    PointVector world(scan.points.size());
    vector<float> ranges(world.size());
    for (size_t i = 0; i < world.size(); i++)
    {
        const PointType &p = scan.points[i];   // lidar frame
        ranges[i] = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        pointBodyToWorld(&p, &world[i]);
    }
    double st_time = omp_get_wtime();
    map_new = voxel_map.insert(world, ranges);
    map_candidates = add_point_size = world.size();
    map_rejected = 0;
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

void map_incremental()
{
    if (use_voxel_planes)
        voxel_map_incremental();
    if (!use_ikdtree)
        return;
    const double voxel_time = use_voxel_planes ? kdtree_incremental_time : 0.0;
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    map_new = map_rejected = map_candidates = 0;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* hybrid: a point matched to a voxel plane had no neighbour search (Nearest_Points is stale), and its
           region is already in the ikd-Tree (the plane's first points went through the 5-NN path) */
        if (plane_from_voxel[i])
            continue;
        /* decide if need add to map: the map keeps about one point per filter_size_map voxel, the one
           closest to the voxel centre. Only the point's 5 nearest map neighbours that lie in its own
           voxel are considered:
           - none: the voxel is (almost certainly) empty, add the point without downsampling;
           - one is closer to the voxel centre: the point is not needed;
           - otherwise: add it with downsampling, which keeps the voxel's point closest to the centre. */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            const double half = 0.5 * filter_size_map_min;
            PointType mid_point;
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + half;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + half;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + half;
            const float dist = calc_dist(feats_down_world->points[i], mid_point);
            bool any_in_voxel = false, need_add = true;
            for (const PointType &q : points_near)
            {
                if (fabs(q.x - mid_point.x) > half || fabs(q.y - mid_point.y) > half || fabs(q.z - mid_point.z) > half)
                    continue;
                any_in_voxel = true;
                if (calc_dist(q, mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (!any_in_voxel)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                map_new++;
            }
            else if (need_add && !map_keep_first_point)
            {
                PointToAdd.push_back(feats_down_world->points[i]);
                map_candidates++;
            }
            else
                map_rejected++;
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
            map_candidates++;
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = voxel_time + omp_get_wtime() - st_time;
}

// Publishes `cloud` as float32 x, y, z, intensity (16 B per point), each point mapped by `to_frame`
// (V3D -> V3D), at the scan end time. Nothing is done when nobody subscribes. Replaces pcl::toROSMsg
// of PointXYZINormal: 48 B per point, with zero normals and uninitialised padding.
template <typename ToFrame>
void publish_xyzi(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub, const PointCloudXYZI &cloud,
                  ToFrame to_frame, const char *frame_id)
{
    if (pub->get_subscription_count() == 0)
        return;
    const size_t n = cloud.points.size();
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = get_ros_time(lidar_end_time);
    msg.header.frame_id = frame_id;
    msg.height = 1;
    msg.width = n;
    const char *names[4] = {"x", "y", "z", "intensity"};
    msg.fields.resize(4);
    for (int k = 0; k < 4; k++)
    {
        msg.fields[k].name = names[k];
        msg.fields[k].offset = 4 * k;
        msg.fields[k].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[k].count = 1;
    }
    msg.is_bigendian = false;
    msg.point_step = 16;
    msg.row_step = 16 * n;
    msg.is_dense = true;
    msg.data.resize(16 * n);
    float *out = reinterpret_cast<float *>(msg.data.data());
    for (const PointType &p : cloud.points)
    {
        const V3D q = to_frame(V3D(p.x, p.y, p.z));
        out[0] = q(0);
        out[1] = q(1);
        out[2] = q(2);
        out[3] = p.intensity;
        out += 4;
    }
    pub->publish(msg);
}

// The scan in camera_init (same transform as pointBodyToWorld).
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if (!scan_pub_en)
        return;
    publish_xyzi(pubLaserCloudFull, dense_pub_en ? *feats_undistort : *feats_down_body, [](const V3D &p_body) {
        return V3D(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);
    }, "camera_init");
}

// The full deskewed scan in the IMU (body) frame.
void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    publish_xyzi(pubLaserCloudFull_body, *feats_undistort, [](const V3D &p_lidar) {
        return V3D(state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I);
    }, "body");
}

// The map as held by the ikd-tree (local map, filter_size_map voxels), in the camera_init frame.
bool save_to_pcd()
{
    if (map_file_path.empty() || (use_ikdtree ? ikdtree.Root_Node == nullptr : voxel_map.empty()))
        return false;
    PointCloudXYZI map_cloud;
    if (!use_ikdtree)
        voxel_map.to_cloud(map_cloud);   // voxel means, intensity = points, normal = plane normal
    else
        ikdtree.flatten(ikdtree.Root_Node, map_cloud.points, NOT_RECORD);
    map_cloud.width = map_cloud.points.size();
    map_cloud.height = 1;
    pcl::PCDWriter pcd_writer;
    return pcd_writer.writeBinary(map_file_path, map_cloud) == 0;
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    // Pose covariance in ROS order: position, then rotation about the x, y, z axes of camera_init.
    // The state is ordered [pos, rot, ...] and its rotation error is a body-frame perturbation
    // (rot * Exp(d)), so the rotation part is turned into camera_init: cov = J P J^T, J = diag(I, R).
    Matrix<double, 6, 6> J = Matrix<double, 6, 6>::Identity();
    J.block<3, 3>(3, 3) = state_point.rot.toRotationMatrix();
    const Matrix<double, 6, 6> pose_cov = J * kf.get_P().block<6, 6>(0, 0) * J.transpose();
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            odomAftMapped.pose.covariance[i * 6 + j] = pose_cov(i, j);
    pubOdomAftMapped->publish(odomAftMapped);
    odom_count++;

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";
    trans.child_frame_id = "body";
    trans.header.stamp = get_ros_time(lidar_end_time);
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        // The whole path goes out each time (O(length): ~0.4 ms per publish on the Mac after 18 min), so
        // only when someone listens; a later subscriber still gets the full path.
        if (pubPath->get_subscription_count() > 0)
            pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        #pragma omp parallel num_threads(omp_threads)
    #endif
    {
    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);  // one per thread, reused for every point
    #ifdef MP_EN
        #pragma omp for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
            plane_from_voxel[i] = use_voxel_planes && voxel_map.plane_for(p_global, plane_coef[i]);
        if (ekfom_data.converge && plane_from_voxel[i])
        {
            /** A valid plane of the point's voxel or a neighbouring one **/
            plane_ok[i] = true;
            point_selected_surf[i] = true;
            point_status[i] = PT_MATCHED;
        }
        else if (ekfom_data.converge && !use_ikdtree)
        {
            /** Voxel map only and no valid plane (status "no neighbours") **/
            plane_ok[i] = false;
            point_selected_surf[i] = false;
            point_status[i] = PT_NO_NEIGHBOURS;
        }
        else if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
            point_status[i] = point_selected_surf[i] ? PT_MATCHED : points_near.size() < NUM_MATCH_POINTS ? PT_NO_NEIGHBOURS : PT_FAR_NEIGHBOURS;
        }

        if (!point_selected_surf[i]) continue;

        const VF(4) &pabcd = plane_coef[i];
        point_selected_surf[i] = false;
        point_status[i] = PT_PLANE_FAIL;
        if (ekfom_data.converge && !plane_from_voxel[i])
            plane_ok[i] = esti_plane(plane_coef[i], points_near, 0.1f);
        if (plane_ok[i])
        {
            point_status[i] = PT_RESIDUAL_FAIL;
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                point_status[i] = PT_MATCHED;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    }
    
    effct_feat_num = 0;
    update_calls++;
    std::fill(match_counts, match_counts + 5, 0);
    voxel_matched = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        match_counts[point_status[i]]++;
        voxel_matched += point_selected_surf[i] && plane_from_voxel[i];
    }

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points.push_back(feats_down_body->points[i]);
            corr_normvect->points.push_back(normvec->points[i]);
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x.resize(effct_feat_num, 12);  // every row is written in full below, no zero fill needed
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<int>("num_threads", MP_PROC_NUM);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.lidar_reliable", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<string>("mapping.map_insert_rule", "closest_to_centre");
        this->declare_parameter<string>("map_type", "ikdtree");
        this->declare_parameter<double>("voxel_map.size", 0.25);
        this->declare_parameter<int>("voxel_map.min_points", 20);
        this->declare_parameter<double>("voxel_map.max_thickness", 0.03);
        this->declare_parameter<double>("voxel_map.planarity", 0.05);
        this->declare_parameter<int>("voxel_map.max_points", 30);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", LIVOX_CUSTOM);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<string>("preprocess.time_field", "time");
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<bool>("preprocess.time_absolute", false);
        this->declare_parameter<double>("preprocess.point_keep_ratio", 1.0);
        this->declare_parameter<int>("point_filter_num", 0);   // deprecated: keep ratio 1/N
        this->declare_parameter<int>("random_seed", 0);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<string>("runtime_pos_log_path", "");
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<int>("num_threads", omp_threads, MP_PROC_NUM);
        omp_threads = max(omp_threads, 1);
#ifdef MP_EN
        RCLCPP_INFO(this->get_logger(), "Matching loop: %d OpenMP thread(s)", omp_threads);
#else
        RCLCPP_INFO(this->get_logger(), "Matching loop: single-threaded (built without OpenMP)");
#endif
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        bool lidar_reliable = false;
        this->get_parameter_or<bool>("common.lidar_reliable", lidar_reliable, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        string map_insert_rule;
        this->get_parameter_or<string>("mapping.map_insert_rule", map_insert_rule, "closest_to_centre");
        if (map_insert_rule != "closest_to_centre" && map_insert_rule != "first_point")
        {
            RCLCPP_FATAL(this->get_logger(), "Unknown mapping.map_insert_rule '%s' (closest_to_centre, first_point)",
                         map_insert_rule.c_str());
            throw std::invalid_argument("mapping.map_insert_rule");
        }
        map_keep_first_point = map_insert_rule == "first_point";
        string map_type;
        this->get_parameter_or<string>("map_type", map_type, "ikdtree");
        if (map_type != "ikdtree" && map_type != "voxel" && map_type != "hybrid")
        {
            RCLCPP_FATAL(this->get_logger(), "Unknown map_type '%s' (ikdtree, voxel, hybrid)", map_type.c_str());
            throw std::invalid_argument("map_type");
        }
        use_ikdtree = map_type != "voxel";
        use_voxel_planes = map_type != "ikdtree";
        VoxelMap::Params vp;
        this->get_parameter_or<double>("voxel_map.size", vp.size, 0.25);
        this->get_parameter_or<int>("voxel_map.min_points", vp.min_points, 20);
        this->get_parameter_or<double>("voxel_map.max_thickness", vp.max_thickness, 0.03);
        this->get_parameter_or<double>("voxel_map.planarity", vp.planarity, 0.05);
        this->get_parameter_or<int>("voxel_map.max_points", vp.max_points, 30);
        voxel_map.set_params(vp);
        if (use_voxel_planes)
            RCLCPP_INFO(this->get_logger(), "Map: %s, voxel planes, size %.2f m, min %d points, thickness <= %.3f m, planarity <= %.2f, frozen at %d points",
                        map_type.c_str(), vp.size, vp.min_points, vp.max_thickness, vp.planarity, vp.max_points);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, LIVOX_CUSTOM);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<string>("preprocess.time_field", p_pre->time_field, "time");
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<bool>("preprocess.time_absolute", p_pre->time_absolute, false);
        this->get_parameter_or<double>("preprocess.point_keep_ratio", p_pre->point_keep_ratio, 1.0);
        int random_seed = 0;
        this->get_parameter_or<int>("random_seed", random_seed, 0);
        p_pre->random_seed = uint64_t(random_seed);
        int point_filter_num = 0;
        this->get_parameter_or<int>("point_filter_num", point_filter_num, 0);
        if (point_filter_num > 1 && p_pre->point_keep_ratio >= 1.0)
        {
            p_pre->point_keep_ratio = 1.0 / point_filter_num;
            RCLCPP_WARN(this->get_logger(), "point_filter_num is deprecated (it kept a fixed pattern of points): "
                        "using preprocess.point_keep_ratio = 1/%d", point_filter_num);
        }
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<string>("runtime_pos_log_path", time_log_path, "");
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());

        if (p_pre->lidar_type == LIVOX_CUSTOM)
            RCLCPP_INFO(this->get_logger(), "Lidar input: livox_ros_driver2/CustomMsg on %s", lid_topic.c_str());
        else
            RCLCPP_INFO(this->get_logger(), "Lidar input: PointCloud2 on %s, per-point time field '%s' (unit %d, %s)",
                        lid_topic.c_str(), p_pre->time_field.c_str(), p_pre->time_unit,
                        p_pre->time_absolute ? "absolute" : "after the header stamp");

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id ="camera_init";

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** ROS subscribe initialization ***/
        if (p_pre->lidar_type == LIVOX_CUSTOM)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else
        {
            // A Mid-360 scan is ~440 KB, i.e. hundreds of UDP fragments. Best effort loses the whole
            // scan if one fragment is dropped (e.g. a full socket buffer; the G1's rmem_max is 212 KB);
            // reliable re-sends the fragment instead. Safe against keep-last publishers (the G1 vendor
            // stream is keep-last, depth 1): a slow reader cannot block them.
            const auto lidar_qos = lidar_reliable ? rclcpp::QoS(rclcpp::KeepLast(10)).reliable()
                                                  : rclcpp::QoS(rclcpp::SensorDataQoS());
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, lidar_qos, standard_pcl_cbk);
        }
        // Depth 200 (1 s at 200 Hz): callbacks and scan processing share one thread, so with a
        // depth of 10 any scan taking longer than 50 ms would drop IMU samples.
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 200, imu_cbk);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

public:
    // Processes the next scan once IMU data past its end has arrived (sync_packages). Returns false if
    // there was nothing to process. Called by the 100 Hz timer, or directly by the offline replay.
    bool process_next_scan()
    {
        if (!sync_packages(Measures))
            return false;
        if (flg_first_scan)
        {
            first_lidar_time = Measures.lidar_beg_time;
            flg_first_scan = false;
            return true;
        }

        double t0, t1, t3, t5;

        match_time = 0;
        kdtree_search_time = 0.0;
        solve_time = 0;
        t0 = omp_get_wtime();

        p_imu->Process(Measures, kf, feats_undistort);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
            return true;
        }

        flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                        false : true;
        /*** Segment the map in lidar FOV (ikd-Tree only) ***/
        if (use_ikdtree)
            lasermap_fov_segment();

        /*** downsample the feature points in a scan ***/
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        t1 = omp_get_wtime();
        feats_down_size = feats_down_body->points.size();
        /*** initialize the maps with the first scan ***/
        const bool init_voxel = use_voxel_planes && voxel_map.empty();
        if (init_voxel)
        {
            RCLCPP_INFO(this->get_logger(), "Initialize the voxel map");
            voxel_map_incremental();
            if (!use_ikdtree)
                return true;
        }
        if(use_ikdtree && ikdtree.Root_Node == nullptr)
        {
            RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
            if(feats_down_size > 5)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for(int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                ikdtree.Build(feats_down_world->points);
            }
            return true;
        }
        kdtree_size_st = use_ikdtree ? ikdtree.size() : voxel_map.size();

        /*** ICP and iterated Kalman filter update ***/
        if (feats_down_size < 5)
        {
            RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
            return true;
        }

        normvec->resize(feats_down_size);
        feats_down_world->resize(feats_down_size);
        Nearest_Points.resize(feats_down_size);
        point_selected_surf.resize(feats_down_size);
        res_last.resize(feats_down_size);
        plane_coef.resize(feats_down_size);
        plane_ok.resize(feats_down_size);
        plane_from_voxel.resize(feats_down_size);
        point_status.resize(feats_down_size);
        update_calls = 0;

        /*** iterated state estimation ***/
        double t_update_start = omp_get_wtime();
        double solve_H_time = 0;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        geoQuat.x = state_point.rot.coeffs()[0];
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];

        double t_update_end = omp_get_wtime();

        /******* Publish odometry *******/
        publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

        /*** add the feature points to map kdtree ***/
        t3 = omp_get_wtime();
        map_incremental();
        t5 = omp_get_wtime();

        /******* Publish points *******/
        if (path_en)                         publish_path(pubPath_);
        if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
        if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);

        /*** Timing log ***/
        if (runtime_pos_log)
        {
            frame_num ++;
            kdtree_size_end = use_ikdtree ? ikdtree.size() : voxel_map.size();
            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
            aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
            aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
            aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
            aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
            time_log.push_back({Measures.lidar_beg_time, t5 - t0, kdtree_incremental_time, kdtree_search_time,
                                kdtree_delete_time, current_preprocess_time, int(feats_undistort->points.size()),
                                kdtree_delete_counter, kdtree_size_st, kdtree_size_end, add_point_size,
                                feats_down_size, {match_counts[0], match_counts[1], match_counts[2], match_counts[3],
                                match_counts[4]}, update_calls, res_mean_last, map_new, map_rejected, map_candidates,
                                voxel_matched});
            printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
        }
        return true;
    }

private:
    void timer_callback()
    {
        process_next_scan();
    }

    void map_save_callback(std_srvs::srv::Trigger::Request::SharedPtr req, std_srvs::srv::Trigger::Response::SharedPtr res)
    {
        (void)req;
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        if (!pcd_save_en)
        {
            res->success = false;
            res->message = "Map save disabled (pcd_save.pcd_save_en).";
            return;
        }
        res->success = save_to_pcd();
        res->message = res->success ? "Map saved." : "Map not saved (no map yet, empty map_file_path, or write error).";
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    int frame_num = 0;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    double epsi[23] = {0.001};
};

void write_time_log()
{
    string path = time_log_path;
    if (path.empty())
    {
        string dir = root_dir + "Log";
        mkdir(dir.c_str(), 0755);             // may already exist
        path = dir + "/fast_lio_time_log.csv";
    }
    FILE *fp = fopen(path.c_str(), "w");
    if (fp == nullptr)
    {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    fprintf(fp, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time, down points, "
                "matched, no neighbours, far neighbours, plane fail, residual fail, update calls, residual mean, "
                "map new, map rejected, map candidates, voxel plane matched\n");
    for (const auto &r : time_log)
        fprintf(fp, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f,%d,%d,%d,%d,%d,%d,%d,%0.6f,%d,%d,%d,%d\n", r.stamp,
                r.total, r.scan_points, r.incremental, r.search, r.delete_size, r.delete_time, r.tree_size_st,
                r.tree_size_end, r.add_points, r.preprocess, r.down_points, r.matched[0], r.matched[1], r.matched[2],
                r.matched[3], r.matched[4], r.update_calls, r.residual_mean, r.map_new, r.map_rejected, r.map_candidates,
                r.voxel_matched);
    fclose(fp);
}

// Live:    fastlio_mapping --ros-args --params-file <config> ...
// Offline: fastlio_mapping --replay <dataset.g1replay> <odom.tum> --ros-args --params-file <config> ...
//          feeds the dataset straight into the callbacks (no DDS, no timer) and exits; see replay.hpp.
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const vector<string> args = rclcpp::remove_ros_arguments(argc, argv);
    auto node = std::make_shared<LaserMappingNode>();
    int ret = 0;
    if (args.size() > 1 && args[1] == "--replay")
    {
#ifndef FASTLIO_REPLAY
        fprintf(stderr, "fastlio_mapping was built without zlib: no --replay\n");
        return 2;
#else
        if (!(args.size() == 4 || (args.size() == 6 && args[4] == "--map")))
        {
            fprintf(stderr, "usage: fastlio_mapping --replay <dataset.g1replay> <odom.tum> [--map <map.pcd>] [--ros-args ...]\n");
            return 2;
        }
        uint64_t poses_taken = 0;
        replay::Hooks hooks;
        hooks.imu = [](sensor_msgs::msg::Imu::UniquePtr m) { imu_cbk(std::move(m)); };
        hooks.lidar = [](sensor_msgs::msg::PointCloud2::UniquePtr m) { standard_pcl_cbk(std::move(m)); };
        hooks.process = [&node]() { return node->process_next_scan(); };
        hooks.ok = []() { return rclcpp::ok(); };
        hooks.new_pose = [&poses_taken](nav_msgs::msg::Odometry &odom) {
            if (odom_count == poses_taken)
                return false;
            poses_taken = odom_count;
            odom = odomAftMapped;
            return true;
        };
        ret = replay::run(args[2], args[3], hooks);
        if (args.size() == 6)
        {
            map_file_path = args[5];
            if (!save_to_pcd())
                fprintf(stderr, "could not write the map to %s\n", map_file_path.c_str());
        }
#endif
    }
    else
    {
        rclcpp::spin(node);
    }
    node.reset();
    if (rclcpp::ok())
        rclcpp::shutdown();
    if (runtime_pos_log)
        write_time_log();
    return ret;
}

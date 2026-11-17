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
#include <filesystem>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <chrono>
#include <cstring>
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
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#ifdef FASTLIO_REPLAY
#include "replay.hpp"
#endif
#include <ikd-Tree/ikd_Tree.h>
#include <voxel_map.h>
#include <scan_patches.h>
#include <imu_restamper.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)

/*** Timing log (runtime_pos_log_enable): one row per processed scan, written at exit ***/
struct TimeLogRow
{
    double stamp, total, preprocess;   // scan start [s], processing and preprocessing time [s]
    int scan_points, down_points;      // after preprocessing, after downsampling
    int map_size;                      // ikd-Tree points or voxels, after the scan
    int matched;                       // down points with a residual (last update iteration)
    int update_calls;                  // update iterations (h_share_model calls)
    double residual_mean;              // mean |residual| of the matched points [m]
    // The state after the update: tilt of the gravity state from -z [deg], IMU roll and pitch in the world
    // frame [deg], accelerometer and gyroscope biases.
    double grav_tilt, roll, pitch, ba[3], bg[3];
    double ext_rot[3];                 // lidar-to-IMU rotation estimate as a rotation vector [deg]
    int no_match, gate_fail, pooled, blob;   // down points without a surface, failing the gate, matched to a pooled
                                             // neighbourhood's surface, to a blob
    // The lidar information of the last update iteration: eigenvalues of H'H over the translation (world
    // frame) and the rotation (body frame), ascending, and the weakest translation direction.
    double lam_t[3], lam_r[3], weak_t[3];
    double free_space_time;            // ray casting and removal (voxel_map.free_space) [s]
    int removed;                       // voxels removed as seen through
};
vector<TimeLogRow> time_log;
string time_log_path;                     // runtime_pos_log_path; empty: fast_lio_time_log.csv in the working directory
deque<double> preprocess_time_buffer;     // preprocessing time of each buffered scan, mirrors lidar_buffer
double current_preprocess_time = 0.0;     // of the scan being processed
double match_time = 0, solve_time = 0;
bool   runtime_pos_log = false, map_save_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;                // the input queues (live: the lidar, IMU and processing threads)
condition_variable sig_buffer;   // live: new input for the processing thread
bool new_input = false;

string map_file_path, lid_topic, imu_topic;
// TF frame names (frames.*): map -> init (identity in SLAM; in localization mode set by the first localization),
// init -> odom (the jumps of the estimate, see init_odom), odom -> imu (the filter, continuous), imu -> lidar (the
// extrinsic), imu -> lidar_level (the lidar origin with the axes of init at the end of the initialisation: level
// while the robot stood, whatever the mounting). init is the filter's frame: gravity-aligned, its origin the IMU
// at the start.
struct FrameNames
{
    string map = "map", init = "init", odom = "odom", imu = "imu", lidar = "lidar", lidar_level = "lidar_level";
} frames;

// A rigid transform: p -> R p + t.
struct Rigid
{
    M3D R = Eye3d;
    V3D t = Zero3d;
    V3D operator*(const V3D &p) const { return R * p + t; }
    Rigid operator*(const Rigid &o) const { return Rigid{R * o.R, R * o.t + t}; }
    Rigid inverse() const { return Rigid{R.transpose(), -(R.transpose() * t)}; }
    bool identity() const { return R == Eye3d && t == Zero3d; }
};

// init -> odom: the jumps of the estimate, collected so that odom -> imu stays continuous: an update that moves
// the pose by more than publish.jump_distance [m] or publish.jump_angle_deg, or a relocalization. Identity until
// the first jump. Written by the processing thread together with the IMU rebase (end_scan), read by the IMU
// thread with the newest state: both under mtx_frames, which also guards map -> init (localization).
Rigid init_odom;
std::mutex mtx_frames;
double jump_distance = 0.25, jump_angle = 3.0 * M_PI / 180;
bool output_in_map = false;   // publish.odometry_parent "map": odometry, scans and trajectory in map, not odom

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0, filter_size_map_min = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0;
int    omp_threads = MP_PROC_NUM;   // num_threads: OpenMP threads of the matching loop
bool   lidar_pushed, flg_first_scan = true, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
// The odometry (publish.odometry_*): one topic, the pose and twist of imu, lidar or lidar_level in odom, always the
// latest estimate. odometry_imu_every 0: once per scan (the pose at the scan end, after the lidar update); N: every
// N-th IMU sample (the last scan's state propagated; a scan's correction shows from the next sample sent on).
enum class OdomFrame { IMU, LIDAR, LIDAR_LEVEL };
OdomFrame odometry_frame = OdomFrame::IMU;
string odometry_topic = "/Odometry";
int    odometry_imu_every = 0;
bool   publish_tf = true;                // the TF tree above; odom -> imu with each odometry message
bool   publish_registered_cloud = false;
string registered_cloud_topic = "/registered_cloud";
bool   publish_map = true;               // the map on map_topic every map_period seconds, while someone subscribes
string map_topic = "/Laser_map";
double map_period = 1.0;
double map_point_step = 0.1;             // voxel map: points this far apart over each voxel's distribution (0: its mean)
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
// Why each downsampled point does or does not give a residual (the debug session records it).
enum PointStatus : uint8_t { PT_MATCHED, PT_NO_NEIGHBOURS, PT_FAR_NEIGHBOURS, PT_PLANE_FAIL, PT_RESIDUAL_FAIL };
vector<uint8_t>      point_status;
int matched_last = 0, update_calls = 0;   // timing log: matched points (last iteration), iterations
// Timing log: the lidar information per direction of the last update iteration (h_share_model): eigenvalues
// of H'H over the translation (world frame) and over the rotation (body frame), ascending, and the
// eigenvector of the weakest translation direction.
V3D lam_t_last = Zero3d, lam_r_last = Zero3d, weak_t_last = Zero3d;
// map_type: "ikdtree" (FAST-LIO's map: points, a 5-NN plane fit per query) or "voxel" (voxel_map.h: the
// best-fitting voxel distribution, d2d for a scan patch and point-to-distribution for a single point,
// otherwise a 5-NN plane fit on the points the voxels store).
bool use_ikdtree = true;
VoxelMap voxel_map;
// Per down point (map_type voxel): matched to a voxel distribution (no 5-NN search), that distribution
// (mean, covariance, effective sample size) and where it came from (the debug session records it).
vector<uint8_t> dist_matched;
vector<V3D> dist_mean;
vector<M3D> dist_cov;      // the distribution's covariance, with the uncertainty of its mean
vector<double> dist_neff;  // its effective sample size
vector<double> dist_varmean;   // the uncertainty of its mean
vector<M3D> prior_cov;     // the prior's covariance of the world point (pose uncertainty, see prior_gate)
vector<uint8_t> from_pooled, from_blob;   // the distribution was the pooled neighbourhood's surface / its blob
// mapping.blob_if_below: blob matches (a pooled neighbourhood that is no surface, see VoxelMap) are used only when
// fewer than this share of the scan's down points match a surface: a lifeline where the map is sparse (far
// returns outdoors), not an extra pull where it is not.
double blob_if_below = 0.5;
// mapping.prior_gate: the association and the chi-square gate of a measurement use its covariance plus the
// prior's covariance of the point (the propagated pose uncertainty, mapped to the point: the innovation
// covariance), so a poor prior (an impact with the accelerometer clipped, a direction nothing constrains) widens
// the gate along the uncertain directions instead of losing the matches. The residual rows keep the
// measurement covariance alone, as the Kalman update accounts for the prior itself.
bool prior_gate = true;
// scan_patch.min_points > 0 (scan_patches.h): a scan voxel with at least min_points points is one measurement,
// its centroid with the points' covariance; sparser voxels keep all their points.
int scan_patch_min_points = 0;
// A measurement's covariance (scan side): R (S + σ²/n I) R' + floor² I: the spread S of the patch's n points
// (0 for a single point) plus the noise σ = voxel_map.point_sigma of their centroid, and a systematic floor
// (scan_patch.sigma_floor) that no centroid is trusted beyond. The same formula for every n.
double patch_sigma_floor = 0.02;
const double D2D_GATE = 11.34;                 // squared Mahalanobis distance: chi-square, 3 dof, 99 %
const double D2D_GATE_1 = 6.63;                //   and with 1 dof (a planar voxel's normal row)
// scan_patch.planar_ratio > 0: a matched voxel distribution whose smallest eigenvalue is below this ratio of the
// middle one is a plane: the measurement is one row along its normal (point to plane), not three (its in-plane
// components only say where the voxel grid cut the surface). 0: always three rows.
double planar_ratio = 0.0;
int no_match_last = 0, gate_fail_last = 0, pooled_last = 0, blob_last = 0;   // timing log: per down point, last iteration
vector<ScanPatch> scan_patches;               // per down point (n = 1: a single point)
int scan_patch_count = 0;                     // patch centroids among the down points (debug session)
// Per down point: residual rows (1, or 3 for a distribution match) in normvec[3 i + k].
vector<uint8_t> row_count;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
// Bounded input queues: if processing falls behind or one sensor stops, the oldest data goes first.
// Live: only the newest scan waits (a driver publishes a scan when it is complete, so an older one still
// queued is stale) and 1 s of IMU (200 Hz, ImuProcess). The offline replay feeds each scan before the IMU
// that completes the previous one and must process every scan: main raises the limits.
size_t max_lidar_buffer = 1;

PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
// Offline (--dump-scans): every scan as it went into the voxel map, for analyses outside FAST_LIO. One record per
// scan: lidar_end_time (double), the lidar pose in init at the scan end (R row-major, t: 12 doubles), n (uint32),
// then n x (x, y, z in the lidar frame at the scan end: float32; flags: uint8, bit 0 = inserted into the map, bit 1 =
// kept out as lying in free space), then
// the voxels removed as seen through before the insertion: m (uint32), m x key (uint64, VoxelMap::pack).
FILE *scan_dump = nullptr;
vector<uint8_t> point_flags;   // per point of feats_undistort, as dumped
double free_space_time = 0;    // this scan's ray casting and removal [s]
int removed_last = 0;          // voxels removed in this scan
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
uint64_t odom_count = 0;                  // scans with an updated pose so far (the replay's TUM output)

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

// Localization mode on a saved map (localization.map_file): works on the globals above.
#include "localization.hpp"

// The frame the odometry, the scans and the trajectory are published in, as a transform from init (io: init ->
// odom): odom (publish.odometry_parent "odom", continuous) or map. False while map is not known (localization
// mode before the first localization).
bool output_frame(const Rigid &io, Rigid &T, const string *&name)
{
    if (output_in_map)
    {
        if (loc::state.enabled && !loc::state.localized)
            return false;
        T = loc::state.enabled ? loc::state.map_init : Rigid();
        name = &frames.map;
        return true;
    }
    T = io.inverse();
    name = &frames.odom;
    return true;
}

// Moves the state's pose and velocity by T, and turns its covariance with them.
void move_state(const Rigid &T, state_ikfom &x, Filter::cov &P)
{
    if (T.identity())
        return;
    x.pos = T.R * x.pos + T.t;
    x.rot = SO3(T.R * x.rot.toRotationMatrix());
    x.vel = T.R * x.vel;
    Filter::cov J = Filter::cov::Identity();
    J.block<3, 3>(0, 0) = T.R;
    J.block<3, 3>(12, 12) = T.R;
    P = J * P * J.transpose();
}

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
    // Drains ikd-tree's list of deleted points, which otherwise grows forever.
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
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
    if(cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm);
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
    if (lidar_buffer.size() > max_lidar_buffer)
    {
        static chrono::steady_clock::time_point warned;
        if (warn_now(warned))
            std::cerr << "Processing behind the lidar: dropping unprocessed scans (the newest is kept)" << std::endl;
        drop_queued_scans(max_lidar_buffer);
    }
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg)
{
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    const double preprocess_time = omp_get_wtime() - preprocess_start_time;

    mtx_buffer.lock();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        drop_queued_scans(0);
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;
    enqueue_scan(ptr, cur_time, preprocess_time);
    new_input = true;
    mtx_buffer.unlock();
    sig_buffer.notify_one();
}

void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg)
{
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    const double preprocess_time = omp_get_wtime() - preprocess_start_time;

    mtx_buffer.lock();
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

    if (abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && p_imu->samples() > 0 && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }
    enqueue_scan(ptr, last_timestamp_lidar, preprocess_time);
    new_input = true;
    mtx_buffer.unlock();
    sig_buffer.notify_one();
}

// One IMU sample with its final stamp: queued and integrated.
void add_imu_sample(const sensor_msgs::msg::Imu::SharedPtr &msg)
{
    // Lidar-to-IMU time offset, applied in integer nanoseconds so that the stamp stays exact.
    if (time_diff_lidar_to_imu != 0.0)
        msg->header.stamp = rclcpp::Time(msg->header.stamp) -
                            rclcpp::Duration(std::chrono::nanoseconds(llround(time_diff_lidar_to_imu * 1e9)));

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "IMU loop back, clear buffer" << std::endl;
        p_imu->clear_samples();
    }

    last_timestamp_imu = timestamp;

    if (!p_imu->add_imu(msg))   // integrated at once
    {
        static chrono::steady_clock::time_point warned;
        if (warn_now(warned))
            std::cerr << "IMU queue full (no lidar scans?): dropping the oldest samples" << std::endl;
    }
    const bool scan_waiting = !lidar_buffer.empty();   // only then can this sample complete a scan
    new_input = new_input || scan_waiting;
    mtx_buffer.unlock();
    if (scan_waiting)
        sig_buffer.notify_one();
}

// imu_restamp.*: the IMU stamps rebuilt on the sensor's own sample grid (imu_restamper.h) before anything else
// sees them. Only the IMU callback thread uses it.
struct ImuRestamp
{
    std::unique_ptr<ImuRestamper<sensor_msgs::msg::Imu::SharedPtr>> restamper;
    int64_t epoch_ns = 0, last_ns = 0;   // first raw stamp (the restamper's time origin), previous raw stamp
    double next_log = 0;                 // [s since epoch] next counters line
    double log_period = 30.0;

    void enable(const ImuRestampParams &p)
    {
        restamper = std::make_unique<ImuRestamper<sensor_msgs::msg::Imu::SharedPtr>>(
            p, [this](double t, const sensor_msgs::msg::Imu::SharedPtr &m) {
                const int64_t ns = epoch_ns + llround(t * 1e9);
                m->header.stamp.sec = int32_t(ns / 1000000000);
                m->header.stamp.nanosec = uint32_t(ns % 1000000000);
                add_imu_sample(m);
            });
    }

    void push(const sensor_msgs::msg::Imu::SharedPtr &m)
    {
        const int64_t ns = int64_t(m->header.stamp.sec) * 1000000000 + m->header.stamp.nanosec;
        if (restamper->stats().received == 0)
        {
            epoch_ns = last_ns = ns;
            next_log = log_period;
        }
        const auto &a = m->linear_acceleration;
        const auto &w = m->angular_velocity;
        const uint64_t restarts = restamper->stats().restarts;
        const double s = 1e-9 * double(ns - epoch_ns);
        restamper->push(s, {a.x, a.y, a.z, w.x, w.y, w.z}, m);
        if (restamper->stats().restarts != restarts)
            fprintf(stderr, "IMU restamp: stamp jump of %+.3f s, grid restarted\n", 1e-9 * double(ns - last_ns));
        last_ns = ns;
        if (s >= next_log)
        {
            log();
            next_log = s + log_period;
        }
    }

    void log() const
    {
        const ImuRestampStats &st = restamper->stats();
        fprintf(stderr, "IMU restamp: in %llu out %llu | held %llu, max delay %.1f ms | lost %llu (%llu events) | "
                        "repeats %llu | restarts %llu\n",
                (unsigned long long)st.received, (unsigned long long)st.emitted, (unsigned long long)st.held,
                1e3 * st.max_delay, (unsigned long long)st.lost, (unsigned long long)st.loss_events,
                (unsigned long long)st.duplicates, (unsigned long long)st.restarts);
    }
} imu_restamp;

void imu_cbk(sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    sensor_msgs::msg::Imu::SharedPtr msg(std::move(msg_in));
    if (imu_restamp.restamper)
        imu_restamp.push(msg);
    else
        add_imu_sample(msg);
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
// The next scan, once IMU data past its end has arrived (caller holds mtx_buffer).
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || p_imu->samples() == 0) {
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

void dump_scan(const M3D &R_wl, const V3D &t_wl)
{
    const PointCloudXYZI &scan = *feats_undistort;
    const uint32_t n = scan.points.size();
    double pose[12];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            pose[3 * r + c] = R_wl(r, c);
    for (int r = 0; r < 3; r++)
        pose[9 + r] = t_wl(r);
    fwrite(&lidar_end_time, sizeof(double), 1, scan_dump);
    fwrite(pose, sizeof(double), 12, scan_dump);
    fwrite(&n, sizeof(n), 1, scan_dump);
    vector<char> buf(size_t(n) * 13);
    for (uint32_t i = 0; i < n; i++)
    {
        const float xyz[3] = {scan.points[i].x, scan.points[i].y, scan.points[i].z};
        memcpy(&buf[13 * i], xyz, 12);
        buf[13 * i + 12] = char(point_flags[i]);
    }
    fwrite(buf.data(), 1, buf.size(), scan_dump);
    const vector<uint64_t> &removed = voxel_map.removed();
    const uint32_t m = removed.size();
    fwrite(&m, sizeof(m), 1, scan_dump);
    fwrite(removed.data(), sizeof(uint64_t), m, scan_dump);
}

// Voxel map: every kept, deskewed point of the scan goes into the statistics of its voxel.
void voxel_map_incremental()
{
    const PointCloudXYZI &scan = *feats_undistort;
    point_flags.assign(scan.points.size(), 0);
    const M3D R_wl = state_point.rot.toRotationMatrix() * state_point.offset_R_L_I.toRotationMatrix();
    const V3D t_wl = state_point.rot * state_point.offset_T_L_I + state_point.pos;
    vector<V3D> q(scan.points.size());
    for (size_t i = 0; i < q.size(); i++)
        q[i] = R_wl * V3D(scan.points[i].x, scan.points[i].y, scan.points[i].z) + t_wl;
    // What moved (voxel_map.free_space): every point is a hit in the free-space cells, and a point that fell where
    // the earlier rays found free space is kept out (observe); rays to points of the whole scan (also those the
    // primary map covers in localization) clear the voxels they see through, the inserted points confirm the voxels
    // they fit (carve, insert, settle).
    const double t0 = omp_get_wtime();
    vector<uint8_t> in_free_space(q.size());
    for (size_t i = 0; i < q.size(); i++)
        in_free_space[i] = voxel_map.observe(q[i]);
    const double t1 = omp_get_wtime();
    PointVector world(q.size());
    vector<float> ranges(q.size());
    size_t n = 0;
    for (size_t i = 0; i < q.size(); i++)
    {
        const PointType &p = scan.points[i];   // lidar frame
        if (!loc::to_live_layer(q[i]))
            continue;   // localization: the primary map covers this place
        if (in_free_space[i])
        {
            point_flags[i] = 2;
            continue;
        }
        point_flags[i] = 1;
        ranges[n] = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        world[n].x = q[i](0);
        world[n].y = q[i](1);
        world[n].z = q[i](2);
        world[n].intensity = p.intensity;
        n++;
    }
    world.resize(n);
    ranges.resize(n);
    const double t2 = omp_get_wtime();
    voxel_map.carve(R_wl, t_wl, scan.points);
    const double t3 = omp_get_wtime();
    voxel_map.insert(world, ranges);
    const double t4 = omp_get_wtime();
    removed_last = voxel_map.settle();
    free_space_time = (t1 - t0) + (t3 - t2) + (omp_get_wtime() - t4);
    if (scan_dump != nullptr)
        dump_scan(R_wl, t_wl);
}

void map_incremental()
{
    if (!use_ikdtree)
    {
        voxel_map_incremental();
        return;
    }
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
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
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
            else if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
            PointToAdd.push_back(feats_down_world->points[i]);
    }
    ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

// Publishes `cloud` as float32 x, y, z, intensity (16 B per point), each point mapped by `to_frame`
// (V3D -> V3D), at the scan end time. Nothing is done when nobody subscribes.
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

// A map as float32 x, y, z, intensity and rgb (20 B per point), mapped by T. Voxel map (surfaces): the color is the
// direction of the plane normal in the output frame, muted so it reads on white and black backgrounds: walls facing
// x red, facing y green, floors and ceilings blue-gray, slopes in between (not one surface: gray, darker than the
// floor), the intensity its vertical part |n_z| (not one surface: 0.5). ikd-Tree: gray, the points' intensity. rgb
// is packed as PCL does (Foxglove: color mode "RGB"; RViz: "RGB8").
void map_cloud_msg(const PointCloudXYZI &cloud, const Rigid &T, bool surfaces, sensor_msgs::msg::PointCloud2 &msg)
{
    const size_t n = cloud.points.size();
    msg.height = 1;
    msg.width = n;
    const char *names[5] = {"x", "y", "z", "intensity", "rgb"};
    msg.fields.resize(5);
    for (int k = 0; k < 5; k++)
    {
        msg.fields[k].name = names[k];
        msg.fields[k].offset = 4 * k;
        msg.fields[k].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[k].count = 1;
    }
    msg.is_bigendian = false;
    msg.point_step = 20;
    msg.row_step = 20 * n;
    msg.is_dense = true;
    msg.data.resize(20 * n);
    float *out = reinterpret_cast<float *>(msg.data.data());
    for (const PointType &p : cloud.points)
    {
        const V3D q = T * V3D(p.x, p.y, p.z);
        const V3D normal = T.R * V3D(p.normal_x, p.normal_y, p.normal_z);
        const bool planar = surfaces && normal.squaredNorm() > 0.5;
        uint32_t rgb = 0xa0a0a0;
        if (planar)
        {
            // floor color blended toward the wall color by the horizontal part squared, the wall color by the
            // wall's direction (|n_x| : |n_y|)
            static const V3D floor(0.66, 0.70, 0.80);
            const double h = std::hypot(normal(0), normal(1));
            V3D wall(0.35, 0.35, 0.40);
            if (h > 1e-9)
                wall += V3D(0.55 * std::abs(normal(0)) / h, 0.55 * std::abs(normal(1)) / h, 0);
            const V3D c = floor + h * h * (wall - floor);
            rgb = uint32_t(255 * c(0) + 0.5) << 16 | uint32_t(255 * c(1) + 0.5) << 8 | uint32_t(255 * c(2) + 0.5);
        }
        out[0] = q(0);
        out[1] = q(1);
        out[2] = q(2);
        out[3] = !surfaces ? p.intensity : planar ? std::abs(normal(2)) : 0.5;
        std::memcpy(&out[4], &rgb, 4);
        out += 5;
    }
}

// A deskewed scan point (lidar frame) in the world frame (same transform as pointBodyToWorld).
V3D lidar_to_world(const V3D &p_lidar)
{
    return V3D(state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos);
}

// The scan in the output frame (T: from init).
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull,
                         const Rigid &T, const string &frame)
{
    if (!scan_pub_en)
        return;
    publish_xyzi(pubLaserCloudFull, dense_pub_en ? *feats_undistort : *feats_down_body,
                 [&T](const V3D &p) { return T * lidar_to_world(p); }, frame.c_str());
}

// The map in the output frame (T: from init), every map_period seconds of scan time (0: every scan) while someone
// subscribes: the voxel map's distributions as points map_point_step apart (in localization the live layer, the
// saved map is /prior_map) or the ikd-Tree's points; colors as map_cloud_msg. Called by the lidar thread after the
// map update, so the map does not change meanwhile.
void publish_map_cloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pub, const Rigid &T,
                 const string &frame)
{
    static double last = -1;
    if (pub->get_subscription_count() == 0 ||
        (last >= 0 && lidar_end_time >= last && lidar_end_time - last < map_period))   // earlier: a replay restarted
        return;
    last = lidar_end_time;
    PointCloudXYZI map_cloud;
    if (use_ikdtree)
    {
        if (ikdtree.Root_Node == nullptr)
            return;
        ikdtree.flatten(ikdtree.Root_Node, map_cloud.points, NOT_RECORD);
    }
    else
        voxel_map.to_cloud(map_cloud, map_point_step);
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp = get_ros_time(lidar_end_time);
    msg.header.frame_id = frame;
    map_cloud_msg(map_cloud, T, !use_ikdtree, msg);
    pub->publish(msg);
}

// The full deskewed scan in the IMU (body) frame.
void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    publish_xyzi(pubLaserCloudFull_body, *feats_undistort, [](const V3D &p_lidar) {
        return V3D(state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I);
    }, frames.imu.c_str());
}

// A leading ~/ is the home directory: map paths in a config (map_file_path, localization.map_file).
string expand_home(const string &path)
{
    const char *home = getenv("HOME");
    return path.rfind("~/", 0) == 0 && home != nullptr ? string(home) + path.substr(1) : path;
}

// The rotation that levels a SLAM map: the gravity estimate onto -z, the smallest such rotation (the heading is
// kept). init is level only as the IMU found it while the robot stood (the mean specific force, tilted by an
// accelerometer bias); the gravity state corrects that once the robot has moved.
M3D map_leveling()
{
    const V3D g = state_point.grav;
    return Eigen::Quaterniond::FromTwoVectors(g.normalized(), -V3D::UnitZ()).toRotationMatrix();
}

// Writes the map: map_file_path (default fast_lio_map in the working directory; its folder is created) with the map
// type's extension, .vmap (the voxel map itself) or .pcd (the ikd-Tree's points). In localization mode the live layer
// (frame init; the file holds map -> init), to <name>_local_map_<n>.vmap if the file exists (the primary map, an
// earlier save): nothing is overwritten. Returns the path written, "" if there is no map yet or the file cannot be
// written.
// A SLAM map is written level (map_leveling: its frame is init rotated by that); leveled_deg (if given) gets the angle
// [deg], 0 for a live layer.
string save_map(double *leveled_deg = nullptr)
{
    const M3D level = loc::state.enabled ? M3D::Identity() : map_leveling();
    if (leveled_deg != nullptr)
        *leveled_deg = Eigen::AngleAxisd(level).angle() * 180 / M_PI;
    if (use_ikdtree ? ikdtree.Root_Node == nullptr : voxel_map.empty())
        return "";
    string stem = map_file_path.empty() ? "fast_lio_map" : map_file_path;
    const size_t slash = stem.find_last_of('/'), dot = stem.find_last_of('.');
    if (dot != string::npos && (slash == string::npos || dot > slash + 1))
        stem.erase(dot);   // the extension follows the map type (not a leading dot: ./name)
    const string ext = use_ikdtree ? ".pcd" : VoxelMap::EXTENSION;
    string path = stem + ext;
    if (loc::state.enabled)
        for (int n = 1; std::filesystem::exists(path); n++)
            path = stem + "_local_map_" + std::to_string(n) + ext;
    const std::filesystem::path folder = std::filesystem::path(path).parent_path();
    std::error_code ec;
    if (!folder.empty())
        std::filesystem::create_directories(folder, ec);   // a failure shows as the write failing
    if (use_ikdtree)
    {
        PointCloudXYZI map_cloud;
        ikdtree.flatten(ikdtree.Root_Node, map_cloud.points, NOT_RECORD);
        for (PointType &p : map_cloud.points)
        {
            const V3D q = level * V3D(p.x, p.y, p.z);
            p.x = q(0), p.y = q(1), p.z = q(2);
        }
        map_cloud.width = map_cloud.points.size();
        map_cloud.height = 1;
        pcl::PCDWriter pcd_writer;
        return pcd_writer.writeBinary(path, map_cloud) == 0 ? path : "";
    }
    Eigen::Isometry3d frame = Eigen::Isometry3d::Identity();
    if (loc::state.active())
    {
        frame.linear() = loc::state.map_init.R;
        frame.translation() = loc::state.map_init.t;
    }
    return voxel_map.save(path, frame, level) ? path : "";
}

// Pose of the IMU in the frame `parent` (the state's frame), with its covariance.
void fill_odometry(nav_msgs::msg::Odometry &odom, double stamp, const state_ikfom &x, const Filter::cov &P,
                   const string &parent)
{
    odom.header.frame_id = parent;
    odom.child_frame_id = frames.imu;
    odom.header.stamp = get_ros_time(stamp);
    odom.pose.pose.position.x = x.pos(0);
    odom.pose.pose.position.y = x.pos(1);
    odom.pose.pose.position.z = x.pos(2);
    odom.pose.pose.orientation.x = x.rot.coeffs()[0];
    odom.pose.pose.orientation.y = x.rot.coeffs()[1];
    odom.pose.pose.orientation.z = x.rot.coeffs()[2];
    odom.pose.pose.orientation.w = x.rot.coeffs()[3];
    // Pose covariance in ROS order: position, then rotation about the x, y, z axes of the world frame.
    // The state is ordered [pos, rot, ...] and its rotation error is a body-frame perturbation
    // (rot * Exp(d)), so the rotation part is turned into the world frame: cov = J P J^T, J = diag(I, R).
    Matrix<double, 6, 6> J = Matrix<double, 6, 6>::Identity();
    J.block<3, 3>(3, 3) = x.rot.toRotationMatrix();
    const Matrix<double, 6, 6> pose_cov = J * P.block<6, 6>(0, 0) * J.transpose();
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            odom.pose.covariance[i * 6 + j] = pose_cov(i, j);
}

// The twist of a frame S attached to the IMU (R_IS: S to IMU, t_IS: its origin in the IMU frame), in S:
// velocity and angular rate. The rotation uncertainty is left out of the velocity covariance.
void fill_twist(nav_msgs::msg::Odometry &odom, const state_ikfom &x, const Filter::cov &P, const V3D &angvel,
                const M3D &R_IS, const V3D &t_IS)
{
    const M3D R_WS = x.rot.toRotationMatrix() * R_IS;
    const V3D v = R_WS.transpose() * (x.vel + x.rot * angvel.cross(t_IS));
    const V3D w = R_IS.transpose() * angvel;
    const M3D v_cov = R_WS.transpose() * P.block<3, 3>(12, 12) * R_WS;
    odom.twist.twist.linear.x = v(0);
    odom.twist.twist.linear.y = v(1);
    odom.twist.twist.linear.z = v(2);
    odom.twist.twist.angular.x = w(0);
    odom.twist.twist.angular.y = w(1);
    odom.twist.twist.angular.z = w(2);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            odom.twist.covariance[i * 6 + j] = v_cov(i, j);
}

// The pose and twist of a frame S attached to the IMU (R_IS: S to IMU, t_IS: its origin in the IMU frame) in the
// frame `parent` (the state's frame), with the covariances.
void fill_frame_odometry(nav_msgs::msg::Odometry &odom, double stamp, const state_ikfom &x, const Filter::cov &P,
                         const V3D &angvel, const M3D &R_IS, const V3D &t_IS, const string &child,
                         const string &parent)
{
    const M3D R = x.rot.toRotationMatrix();
    const M3D R_WS = R * R_IS;
    const V3D p = x.pos + R * t_IS;
    const Eigen::Quaterniond q(R_WS);
    odom.header.frame_id = parent;
    odom.child_frame_id = child;
    odom.header.stamp = get_ros_time(stamp);
    odom.pose.pose.position.x = p(0);
    odom.pose.pose.position.y = p(1);
    odom.pose.pose.position.z = p(2);
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    // As fill_odometry, with the lever arm: with rot = R Exp(d), p + rot t_IS moves by dp - R [t_IS]x d,
    // and the rotation error in the world frame is R d (the same for every frame attached to the body).
    M3D t_hat;
    t_hat << SKEW_SYM_MATRX(t_IS);
    Matrix<double, 6, 6> J = Matrix<double, 6, 6>::Zero();
    J.block<3, 3>(0, 0) = M3D::Identity();
    J.block<3, 3>(0, 3) = -R * t_hat;
    J.block<3, 3>(3, 3) = R;
    const Matrix<double, 6, 6> pose_cov = J * P.block<6, 6>(0, 0) * J.transpose();
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            odom.pose.covariance[i * 6 + j] = pose_cov(i, j);
    fill_twist(odom, x, P, angvel, R_IS, t_IS);
}

// The trajectory: one pose per 2 scans, the last 5000 (about 17 min at 10 Hz; the whole path goes out
// each time), published only when someone listens.
void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath, const Rigid &T, const string &frame)
{
    static int scans = 0;
    if (++scans % 2 == 0)
    {
        state_ikfom x = state_point;
        Filter::cov P = Filter::cov::Zero();
        move_state(T, x, P);
        geometry_msgs::msg::PoseStamped pose;
        pose.header.stamp = get_ros_time(lidar_end_time);
        pose.header.frame_id = frame;
        pose.pose.position.x = x.pos(0);
        pose.pose.position.y = x.pos(1);
        pose.pose.position.z = x.pos(2);
        pose.pose.orientation.x = x.rot.coeffs()[0];
        pose.pose.orientation.y = x.rot.coeffs()[1];
        pose.pose.orientation.z = x.rot.coeffs()[2];
        pose.pose.orientation.w = x.rot.coeffs()[3];
        path.header.frame_id = frame;
        path.poses.push_back(pose);
        if (path.poses.size() > 5000)
            path.poses.erase(path.poses.begin());
        if (pubPath->get_subscription_count() > 0)
            pubPath->publish(path);
    }
}

// Downsamples the deskewed scan into feats_down_body (filter_size_surf voxels in the lidar frame): one
// centroid per voxel, or patches (scan_patch.min_points > 0). Sets feats_down_size.
void downsample_scan()
{
    if (scan_patch_min_points > 0)
        scan_patch_count = voxelize_scan(*feats_undistort, filter_size_surf_min, scan_patch_min_points,
                                         *feats_down_body, scan_patches);
    else
    {
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);
        scan_patch_count = 0;
    }
    feats_down_size = feats_down_body->points.size();
}

// Sizes the per-point arrays of the update to the downsampled scan.
void resize_per_point_arrays()
{
    normvec->resize(3 * feats_down_size);
    row_count.resize(feats_down_size);
    scan_patches.resize(feats_down_size);   // without patches: all single points (n = 1)
    if (!use_ikdtree)
    {
        dist_matched.resize(feats_down_size);
        dist_mean.resize(feats_down_size);
        dist_cov.resize(feats_down_size);
        dist_neff.resize(feats_down_size);
        dist_varmean.resize(feats_down_size);
        prior_cov.resize(feats_down_size);
        from_pooled.resize(feats_down_size);
        from_blob.resize(feats_down_size);
    }
    feats_down_world->resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);
    point_selected_surf.resize(feats_down_size);
    res_last.resize(feats_down_size);
    plane_coef.resize(feats_down_size);
    plane_ok.resize(feats_down_size);
    point_status.resize(feats_down_size);
}

// The covariance of down point i's measurement in the world frame (see patch_sigma_floor).
M3D measurement_cov(int i, const M3D &R_wl)
{
    const ScanPatch &sp = scan_patches[i];
    return R_wl * (sp.cov + (voxel_map.noise2() / sp.n) * M3D::Identity()) * R_wl.transpose() +
           patch_sigma_floor * patch_sigma_floor * M3D::Identity();
}

// Distribution-to-distribution rows of down point i (world position p) against its matched voxel
// distribution (dist_mean, dist_cov, dist_neff): W (p - mean), W = sqrt(R) L^-1 with C = L L' (Cholesky),
// C = dist_cov + measurement_cov. Each row is a point-to-plane row with "normal" W.row(k), so the update sees
// covariance C. A planar distribution (planar_ratio) gives one row along its normal, a measurement against the
// fitted plane: its variance is the plane's uncertainty (the mean's, σ² / n_eff), the measurement's covariance
// along the normal and the uncertainty of the normal (the scatter about the plane over n_eff λ1) times the
// squared in-plane distance of p from the mean; the voxel's own scatter about its plane does not enter (it is
// the map points' noise, already in the plane's uncertainty). Returns the gate statistic: the squared
// Mahalanobis distance under the row variance plus the prior's covariance of the point (prior_gate).
double set_d2d_rows(int i, const V3D &p, const M3D &R_wl)
{
    const M3D C = dist_cov[i] + measurement_cov(i, R_wl);
    const V3D d = p - dist_mean[i];
    if (planar_ratio > 0 && !from_blob[i])   // a blob is never one plane
    {
        Eigen::SelfAdjointEigenSolver<M3D> es;
        es.computeDirect(dist_cov[i]);
        const V3D l = es.eigenvalues();
        if (l(0) < planar_ratio * l(1))
        {
            // a plane: one row along its normal, with the covariance C projected on it
            const V3D n = es.eigenvectors().col(0);
            const double nd = n.dot(d);
            const double scatter = std::max(l(0) - dist_varmean[i], voxel_map.noise2());   // about the plane: roughness or noise
            const double var = dist_varmean[i] + n.dot((C - dist_cov[i]) * n) +
                               scatter / (dist_neff[i] * l(1)) * (d.squaredNorm() - nd * nd);
            const double w = std::sqrt(LASER_POINT_COV / var);
            row_count[i] = 1;
            PointType &row = normvec->points[3 * i];
            row.x = w * n(0);
            row.y = w * n(1);
            row.z = w * n(2);
            row.intensity = w * nd;
            // the gate (1 dof, scaled to the caller's 3-dof threshold) with the prior's uncertainty along n
            return nd * nd / (var + n.dot(prior_cov[i] * n)) * (D2D_GATE / D2D_GATE_1);
        }
    }
    const M3D W = std::sqrt(LASER_POINT_COV) * inverse_cholesky3(C);
    const V3D r = W * d;
    row_count[i] = 3;
    for (int k = 0; k < 3; k++)
    {
        PointType &row = normvec->points[3 * i + k];
        row.x = W(k, 0);
        row.y = W(k, 1);
        row.z = W(k, 2);
        row.intensity = r(k);
    }
    return (inverse_cholesky3(C + prior_cov[i]) * d).squaredNorm();
}

// The interactive debug session (--replay ... --debug-port): works on the globals above.
#include "debug_session.hpp"

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear();
    total_residual = 0.0;
    const M3D R_wl = s.rot.toRotationMatrix() * s.offset_R_L_I.toRotationMatrix();   // lidar to world
    const V3D t_wl = s.rot * s.offset_T_L_I + s.pos;
    const bool researched = ekfom_data.converge;   // this iteration searches the map (debug session)
    // the prior's pose covariance (position, rotation as a right perturbation of rot), for prior_gate
    const Eigen::Matrix<double, 6, 6> P_pose = kf.get_P().topLeftCorner<6, 6>();
    const M3D R_wi = s.rot.toRotationMatrix();

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
        V3D p_global(R_wl * p_body + t_wl);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge && !use_ikdtree)
        {
            /* voxel map: the best-fitting distribution nearby, if it passes the gate */
            const M3D meas_cov = measurement_cov(i, R_wl);
            if (prior_gate && !use_ikdtree)
            {
                // the world point's uncertainty from the prior: p_w = rot Exp(dth) p_i + pos
                const V3D p_i = s.offset_R_L_I * p_body + s.offset_T_L_I;
                Eigen::Matrix<double, 3, 6> J;
                J.leftCols<3>() = M3D::Identity();
                M3D p_hat;
                p_hat << SKEW_SYM_MATRX(p_i);
                J.rightCols<3>() = -R_wi * p_hat;
                prior_cov[i] = J * P_pose * J.transpose();
            }
            else
                prior_cov[i] = M3D::Zero();
            double d2;
            bool pooled = false, blob = false;
            VoxelMap::Distribution dist;
            dist_matched[i] = loc::match_distribution(p_global, meas_cov + prior_cov[i], dist, d2, &pooled, &blob) &&
                d2 < D2D_GATE;
            dist_mean[i] = dist.mean;
            dist_cov[i] = dist.cov;
            dist_neff[i] = dist.n_eff;
            dist_varmean[i] = dist.var_mean;
            from_pooled[i] = dist_matched[i] && pooled && !blob;
            from_blob[i] = dist_matched[i] && blob;
        }
        if (ekfom_data.converge && !use_ikdtree)
        {
            point_selected_surf[i] = dist_matched[i];
            point_status[i] = dist_matched[i] ? PT_MATCHED : PT_NO_NEIGHBOURS;
        }
        else if (ekfom_data.converge)
        {
            /** ikd-Tree: the 5 nearest map points **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
            point_status[i] = point_selected_surf[i] ? PT_MATCHED : points_near.size() < NUM_MATCH_POINTS ? PT_NO_NEIGHBOURS : PT_FAR_NEIGHBOURS;
        }

        if (!point_selected_surf[i]) continue;

        const VF(4) &pabcd = plane_coef[i];
        point_selected_surf[i] = false;
        point_status[i] = PT_PLANE_FAIL;
        if (!use_ikdtree && dist_matched[i])
        {
            /** Against a voxel's distribution: d2d or point to distribution, gated **/
            point_status[i] = PT_RESIDUAL_FAIL;
            if (set_d2d_rows(i, p_global, R_wl) < D2D_GATE)
            {
                point_selected_surf[i] = true;
                point_status[i] = PT_MATCHED;
                res_last[i] = (p_global - dist_mean[i]).norm();   // 3-D distance (the log's residual mean)
            }
            continue;
        }
        if (ekfom_data.converge)
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
                res_last[i] = abs(pd2);
                row_count[i] = 1;
                PointType &row = normvec->points[3 * i];
                row.x = pabcd(0);
                row.y = pabcd(1);
                row.z = pabcd(2);
                row.intensity = pd2;
            }
        }
    }
    }
    
    effct_feat_num = 0;
    update_calls++;
    int matched_points = 0;
    no_match_last = gate_fail_last = pooled_last = blob_last = 0;
    if (!use_ikdtree)
    {
        // blobs only as a lifeline: dropped when enough of the scan matches a surface
        int surfaces = 0, blobs = 0;
        for (int i = 0; i < feats_down_size; i++)
        {
            surfaces += point_selected_surf[i] && !from_blob[i];
            blobs += point_selected_surf[i] && from_blob[i];
        }
        if (blobs > 0 && surfaces >= blob_if_below * feats_down_size)
            for (int i = 0; i < feats_down_size; i++)
                if (point_selected_surf[i] && from_blob[i])
                {
                    point_selected_surf[i] = false;
                    point_status[i] = PT_NO_NEIGHBOURS;
                }
    }
    for (int i = 0; i < feats_down_size; i++)
    {
        no_match_last += point_status[i] == PT_NO_NEIGHBOURS || point_status[i] == PT_FAR_NEIGHBOURS;
        gate_fail_last += point_status[i] == PT_RESIDUAL_FAIL || point_status[i] == PT_PLANE_FAIL;
        pooled_last += point_selected_surf[i] && !use_ikdtree && from_pooled[i];
        blob_last += point_selected_surf[i] && !use_ikdtree && from_blob[i];
        if (point_selected_surf[i])
        {
            for (int k = 0; k < row_count[i]; k++)
            {
                laserCloudOri->points.push_back(feats_down_body->points[i]);
                corr_normvect->points.push_back(normvec->points[3 * i + k]);
            }
            total_residual += res_last[i];
            effct_feat_num += row_count[i];
            matched_points++;
        }
    }
    matched_last = matched_points;

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        if (dbg::session.recording)
            dbg::session.record_iteration(s, researched);
        return;
    }

    res_mean_last = total_residual / matched_points;
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
    {
        // Per-direction information of this iteration (timing log): H'H over translation and rotation.
        Eigen::Matrix<double, 6, 6> HTH = Eigen::Matrix<double, 6, 6>::Zero();
        HTH.selfadjointView<Eigen::Lower>().rankUpdate(ekfom_data.h_x.leftCols<6>().transpose());
        HTH.triangularView<Eigen::StrictlyUpper>() = HTH.transpose();
        const M3D A_t = HTH.block<3, 3>(0, 0), A_r = HTH.block<3, 3>(3, 3);
        Eigen::SelfAdjointEigenSolver<M3D> es_t(A_t), es_r(A_r);
        lam_t_last = es_t.eigenvalues();
        lam_r_last = es_r.eigenvalues();
        weak_t_last = es_t.eigenvectors().col(0);
    }
    solve_time += omp_get_wtime() - solve_start_;
    if (dbg::session.recording)
        dbg::session.record_iteration(s, researched);
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
        this->declare_parameter<string>("frames.map", frames.map);
        this->declare_parameter<string>("frames.init", frames.init);
        this->declare_parameter<string>("frames.odom", frames.odom);
        this->declare_parameter<string>("frames.imu", frames.imu);
        this->declare_parameter<string>("frames.lidar", frames.lidar);
        this->declare_parameter<string>("frames.lidar_level", frames.lidar_level);
        this->declare_parameter<string>("publish.odometry_topic", "/Odometry");
        this->declare_parameter<string>("publish.odometry_frame", "imu");
        this->declare_parameter<int>("publish.odometry_imu_every", 0);
        this->declare_parameter<string>("publish.odometry_parent", "odom");
        this->declare_parameter<double>("publish.jump_distance", 0.25);
        this->declare_parameter<double>("publish.jump_angle_deg", 3.0);
        this->declare_parameter<string>("localization.map_file", "");
        this->declare_parameter<vector<double>>("localization.initial_pose", vector<double>());
        this->declare_parameter<int>("localization.submap_scans", loc::state.submap_scans);
        this->declare_parameter<double>("localization.live_radius", loc::state.live_radius);
        this->declare_parameter<double>("localization.search_xy", loc::state.params.xy_window);
        this->declare_parameter<double>("localization.search_yaw_deg", loc::state.params.yaw_window);
        this->declare_parameter<double>("localization.search_z", loc::state.params.z_window);
        this->declare_parameter<double>("localization.max_range", loc::state.params.max_range);
        this->declare_parameter<vector<double>>("localization.score_band", {loc::state.params.band_low,
                                                                           loc::state.params.band_high});
        this->declare_parameter<double>("localization.accept_matched", loc::state.params.accept_matched);
        this->declare_parameter<double>("localization.max_ambiguity", loc::state.params.max_ambiguity);
        this->declare_parameter<string>("localization.map_topic", "/prior_map");
        this->declare_parameter<string>("localization.initial_pose_topic", "/initialpose");
        this->declare_parameter<string>("localization.status_topic", "/localization_status");
        this->declare_parameter<bool>("publish.publish_tf", true);
        this->declare_parameter<bool>("publish.publish_registered_cloud", false);
        this->declare_parameter<string>("publish.registered_cloud_topic", "/registered_cloud");
        this->declare_parameter<bool>("publish.publish_map", true);
        this->declare_parameter<string>("publish.map_topic", "/Laser_map");
        this->declare_parameter<double>("publish.map_period", 1.0);
        this->declare_parameter<double>("publish.map_point_step", 0.1);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<int>("num_threads", MP_PROC_NUM);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.lidar_reliable", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<bool>("imu_restamp.enable", false);
        this->declare_parameter<double>("imu_restamp.period", ImuRestampParams().period);
        this->declare_parameter<double>("imu_restamp.creep", ImuRestampParams().creep);
        this->declare_parameter<double>("imu_restamp.max_gap", ImuRestampParams().max_gap);
        this->declare_parameter<bool>("imu_restamp.drop_duplicates", ImuRestampParams().drop_duplicates);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<string>("map_type", "ikdtree");
        this->declare_parameter<double>("voxel_map.size", 0.35);
        this->declare_parameter<double>("voxel_map.point_sigma", 0.02);
        this->declare_parameter<double>("voxel_map.shape_ratio", 1.0);
        this->declare_parameter<double>("voxel_map.min_n_eff", 6.0);
        this->declare_parameter<double>("voxel_map.pooled_thickness", 0.1);
        this->declare_parameter<double>("voxel_map.pooled_planarity", 0.2);
        this->declare_parameter<double>("voxel_map.freeze_sigma", 0.004);
        this->declare_parameter<bool>("voxel_map.free_space.enable", true);
        this->declare_parameter<double>("voxel_map.free_space.ray_share", 0.15);
        this->declare_parameter<double>("voxel_map.free_space.min_range", 0.5);
        this->declare_parameter<double>("voxel_map.free_space.max_range", 8.0);
        this->declare_parameter<double>("voxel_map.free_space.ray_sigma", 0.03);
        this->declare_parameter<double>("voxel_map.free_space.core", 2.0);
        this->declare_parameter<double>("voxel_map.free_space.beyond", 3.0);
        this->declare_parameter<double>("voxel_map.free_space.confirm", 3.0);
        this->declare_parameter<int>("voxel_map.free_space.remove_after", 5);
        this->declare_parameter<int>("voxel_map.free_space.subdivision", 4);
        this->declare_parameter<int>("voxel_map.free_space.cell_free_after", 7);
        this->declare_parameter<int>("voxel_map.free_space.cell_free_after_max", 20);
        this->declare_parameter<int>("voxel_map.free_space.cell_reoccupy_after", 5);
        this->declare_parameter<int>("voxel_map.free_space.cell_hit_weight", 9);
        this->declare_parameter<double>("voxel_map.free_space.margin", 0.2);
        this->declare_parameter<double>("voxel_map.free_space.graze_k", 3.0);
        this->declare_parameter<double>("voxel_map.free_space.band_k", 2.0);
        this->declare_parameter<double>("scan_patch.sigma_floor", 0.02);
        this->declare_parameter<int>("scan_patch.min_points", 0);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("mapping.acc_range", 0.0);
        this->declare_parameter<double>("mapping.acc_satu_cov", 30.0);
        this->declare_parameter<double>("mapping.imu_init_time", 0.0);
        this->declare_parameter<double>("mapping.init_cov_bg", 0.0001);
        this->declare_parameter<double>("mapping.init_cov_ba", 0.001);
        this->declare_parameter<double>("mapping.init_cov_grav", 0.00001);
        this->declare_parameter<bool>("mapping.prior_gate", true);
        this->declare_parameter<double>("mapping.blob_if_below", 0.5);
        this->declare_parameter<double>("scan_patch.planar_ratio", 0.0);
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
        this->declare_parameter<bool>("map_save", false);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());

        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter("frames.map", frames.map);
        this->get_parameter("frames.init", frames.init);
        this->get_parameter("frames.odom", frames.odom);
        this->get_parameter("frames.imu", frames.imu);
        this->get_parameter("frames.lidar", frames.lidar);
        this->get_parameter("frames.lidar_level", frames.lidar_level);
        this->get_parameter_or<string>("publish.odometry_topic", odometry_topic, "/Odometry");
        string odom_frame_role;
        this->get_parameter_or<string>("publish.odometry_frame", odom_frame_role, "imu");
        if (odom_frame_role == "imu")
            odometry_frame = OdomFrame::IMU;
        else if (odom_frame_role == "lidar")
            odometry_frame = OdomFrame::LIDAR;
        else if (odom_frame_role == "lidar_level")
            odometry_frame = OdomFrame::LIDAR_LEVEL;
        else
        {
            RCLCPP_FATAL(this->get_logger(), "Unknown publish.odometry_frame '%s' (imu, lidar, lidar_level)",
                         odom_frame_role.c_str());
            throw std::invalid_argument("publish.odometry_frame");
        }
        this->get_parameter_or<int>("publish.odometry_imu_every", odometry_imu_every, 0);
        string odometry_parent;
        this->get_parameter_or<string>("publish.odometry_parent", odometry_parent, "odom");
        if (odometry_parent != "odom" && odometry_parent != "map")
        {
            RCLCPP_FATAL(this->get_logger(), "Unknown publish.odometry_parent '%s' (odom, map)", odometry_parent.c_str());
            throw std::invalid_argument("publish.odometry_parent");
        }
        output_in_map = odometry_parent == "map";
        this->get_parameter_or<double>("publish.jump_distance", jump_distance, 0.25);
        double jump_angle_deg;
        this->get_parameter_or<double>("publish.jump_angle_deg", jump_angle_deg, 3.0);
        jump_angle = jump_angle_deg * M_PI / 180;
        this->get_parameter_or<bool>("publish.publish_tf", publish_tf, true);
        this->get_parameter_or<bool>("publish.publish_registered_cloud", publish_registered_cloud, false);
        this->get_parameter_or<string>("publish.registered_cloud_topic", registered_cloud_topic, "/registered_cloud");
        this->get_parameter_or<bool>("publish.publish_map", publish_map, true);
        this->get_parameter_or<string>("publish.map_topic", map_topic, "/Laser_map");
        this->get_parameter_or<double>("publish.map_period", map_period, 1.0);
        this->get_parameter_or<double>("publish.map_point_step", map_point_step, 0.1);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<int>("num_threads", omp_threads, MP_PROC_NUM);
        omp_threads = max(omp_threads, 1);
#ifdef MP_EN
        RCLCPP_INFO(this->get_logger(), "Matching loop: %d OpenMP thread(s)", omp_threads);
#else
        RCLCPP_INFO(this->get_logger(), "Matching loop: single-threaded (built without OpenMP)");
#endif
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        map_file_path = expand_home(map_file_path);
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        bool lidar_reliable = false;
        this->get_parameter_or<bool>("common.lidar_reliable", lidar_reliable, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        bool imu_restamp_en = false;
        this->get_parameter_or<bool>("imu_restamp.enable", imu_restamp_en, false);
        if (imu_restamp_en)
        {
            ImuRestampParams rp;
            this->get_parameter_or<double>("imu_restamp.period", rp.period, rp.period);
            this->get_parameter_or<double>("imu_restamp.creep", rp.creep, rp.creep);
            this->get_parameter_or<double>("imu_restamp.max_gap", rp.max_gap, rp.max_gap);
            this->get_parameter_or<bool>("imu_restamp.drop_duplicates", rp.drop_duplicates, rp.drop_duplicates);
            imu_restamp.enable(rp);
            RCLCPP_INFO(this->get_logger(), "IMU restamp: stamps rebuilt on a %.2f ms grid%s", 1e3 * rp.period,
                        rp.drop_duplicates ? ", repeated samples dropped" : "");
        }
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        string map_type;
        this->get_parameter_or<string>("map_type", map_type, "ikdtree");
        if (map_type != "ikdtree" && map_type != "voxel")
        {
            RCLCPP_FATAL(this->get_logger(), "Unknown map_type '%s' (ikdtree, voxel)", map_type.c_str());
            throw std::invalid_argument("map_type");
        }
        use_ikdtree = map_type == "ikdtree";
        VoxelMap::Params vp;
        this->get_parameter_or<double>("voxel_map.size", vp.size, 0.35);
        this->get_parameter_or<double>("voxel_map.point_sigma", vp.point_sigma, 0.02);
        this->get_parameter_or<double>("voxel_map.shape_ratio", vp.shape_ratio, 1.0);
        this->get_parameter_or<double>("voxel_map.min_n_eff", vp.min_n_eff, 6.0);
        this->get_parameter_or<double>("voxel_map.pooled_thickness", vp.pooled_thickness, 0.1);
        this->get_parameter_or<double>("voxel_map.pooled_planarity", vp.pooled_planarity, 0.2);
        this->get_parameter_or<double>("voxel_map.freeze_sigma", vp.freeze_sigma, 0.004);
        this->get_parameter_or<double>("scan_patch.sigma_floor", patch_sigma_floor, 0.02);
        voxel_map.set_params(vp);
        VoxelMap::FreeSpace fs;
        this->get_parameter_or<bool>("voxel_map.free_space.enable", fs.enable, true);
        this->get_parameter_or<double>("voxel_map.free_space.ray_share", fs.ray_share, 0.15);
        this->get_parameter_or<double>("voxel_map.free_space.min_range", fs.min_range, 0.5);
        this->get_parameter_or<double>("voxel_map.free_space.max_range", fs.max_range, 8.0);
        this->get_parameter_or<double>("voxel_map.free_space.ray_sigma", fs.ray_sigma, 0.03);
        this->get_parameter_or<double>("voxel_map.free_space.core", fs.core, 2.0);
        this->get_parameter_or<double>("voxel_map.free_space.beyond", fs.beyond, 3.0);
        this->get_parameter_or<double>("voxel_map.free_space.confirm", fs.confirm, 3.0);
        this->get_parameter_or<int>("voxel_map.free_space.remove_after", fs.remove_after, 5);
        this->get_parameter_or<int>("voxel_map.free_space.subdivision", fs.cell.subdivision, 4);
        this->get_parameter_or<int>("voxel_map.free_space.cell_free_after", fs.cell.free_after, 7);
        this->get_parameter_or<int>("voxel_map.free_space.cell_free_after_max", fs.cell.free_after_max, 20);
        this->get_parameter_or<int>("voxel_map.free_space.cell_reoccupy_after", fs.cell.reoccupy_after, 5);
        this->get_parameter_or<int>("voxel_map.free_space.cell_hit_weight", fs.cell.hit_weight, 9);
        this->get_parameter_or<double>("voxel_map.free_space.margin", fs.cell.margin, 0.2);
        this->get_parameter_or<double>("voxel_map.free_space.graze_k", fs.graze_k, 3.0);
        this->get_parameter_or<double>("voxel_map.free_space.band_k", fs.band_k, 2.0);
        fs.threads = omp_threads;   // the rays are walked on the matching loop's threads (num_threads, read above)
        voxel_map.set_free_space(fs);
        setup_localization();
        if (!use_ikdtree)
            RCLCPP_INFO(this->get_logger(), "Map: voxel, size %.2f m, point noise %.0f mm, matched from a shape ratio of %.1f and "
                        "%.0f effective points, frozen when known to %.1f mm", vp.size, 1e3 * vp.point_sigma, vp.shape_ratio,
                        vp.min_n_eff, 1e3 * vp.freeze_sigma);
        this->get_parameter_or<int>("scan_patch.min_points", scan_patch_min_points, 0);
        if (scan_patch_min_points > 0)
            RCLCPP_INFO(this->get_logger(), "Scan patches: voxels with >= %d points as centroid + covariance",
                        scan_patch_min_points);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        double imu_init_time = 0.0, init_cov_bg = 0.0001, init_cov_ba = 0.001, init_cov_grav = 0.00001;
        this->get_parameter_or<double>("mapping.imu_init_time", imu_init_time, 0.0);
        this->get_parameter_or<double>("mapping.init_cov_bg", init_cov_bg, 0.0001);
        this->get_parameter_or<double>("mapping.init_cov_ba", init_cov_ba, 0.001);
        this->get_parameter_or<double>("mapping.init_cov_grav", init_cov_grav, 0.00001);
        this->get_parameter_or<bool>("mapping.prior_gate", prior_gate, true);
        this->get_parameter_or<double>("mapping.blob_if_below", blob_if_below, 0.5);
        this->get_parameter_or<double>("scan_patch.planar_ratio", planar_ratio, 0.0);
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
        this->get_parameter_or<bool>("map_save", map_save_en, false);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());

        if (p_pre->lidar_type == LIVOX_CUSTOM)
            RCLCPP_INFO(this->get_logger(), "Lidar input: livox_ros_driver2/CustomMsg on %s", lid_topic.c_str());
        else
            RCLCPP_INFO(this->get_logger(), "Lidar input: PointCloud2 on %s, per-point time field '%s' (unit %d, %s)",
                        lid_topic.c_str(), p_pre->time_field.c_str(), p_pre->time_unit,
                        p_pre->time_absolute ? "absolute" : "after the header stamp");

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = frames.odom;

        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
        double acc_range, acc_satu_cov;
        this->get_parameter_or<double>("mapping.acc_range", acc_range, 0.0);
        this->get_parameter_or<double>("mapping.acc_satu_cov", acc_satu_cov, 30.0);
        p_imu->set_acc_saturation(acc_range, acc_satu_cov);
        p_imu->set_init_time(imu_init_time);
        p_imu->set_init_cov(init_cov_bg, init_cov_ba, init_cov_grav);

        // Convergence test of the iterated update: position and rotation only, |dx| < 0.001 m / rad (state
        // order: pos, rot, then the extrinsics, vel, bg, ba, gravity). The residuals depend only on the pose;
        // velocity, biases and gravity follow it linearly, and testing them as well keeps almost every scan
        // at the iteration cap.
        fill(epsi, epsi + 23, std::numeric_limits<double>::infinity());
        fill(epsi, epsi + 6, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** ROS subscribe initialization ***/
        // Three threads (live): the IMU callback integrates each sample and publishes the IMU-rate outputs
        // (its own callback group), the lidar callback preprocesses and queues the scans, and the processing
        // thread (start_processing) takes each scan as soon as the IMU data past its end has arrived.
        // Lidar depth 1: a scan that waited in the queue is stale.
        if (p_pre->lidar_type == LIVOX_CUSTOM)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 1,
                [](livox_ros_driver2::msg::CustomMsg::UniquePtr msg) { livox_pcl_cbk(std::move(msg)); });
        }
        else
        {
            // common.lidar_reliable: a scan of a few hundred KB is hundreds of UDP fragments; best effort
            // loses the whole scan if one fragment is dropped (e.g. a full socket buffer), reliable re-sends
            // the fragment. Safe with keep-last publishers: a slow reader cannot block them.
            const auto lidar_qos = lidar_reliable ? rclcpp::QoS(rclcpp::KeepLast(1)).reliable()
                                                  : rclcpp::QoS(rclcpp::SensorDataQoS()).keep_last(1);
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, lidar_qos,
                [](sensor_msgs::msg::PointCloud2::UniquePtr msg) { standard_pcl_cbk(std::move(msg)); });
        }
        // Depth 200 (1 s at 200 Hz): every sample is needed for the propagation.
        rclcpp::SubscriptionOptions imu_options;
        imu_options.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 200,
            [this](sensor_msgs::msg::Imu::UniquePtr msg) {
                imu_cbk(std::move(msg));
                if (odometry_imu_every > 0)
                    publish_imu_odometry();
            }, imu_options);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubOdom_ = this->create_publisher<nav_msgs::msg::Odometry>(odometry_topic, 20);
        // The trajectory (not /path: navigation stacks use that name for the planned path).
        if (path_en)
            pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/trajectory", 20);
        if (publish_tf)
        {
            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
            tf_static_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
            const auto now = this->get_clock()->now();
            std::vector<geometry_msgs::msg::TransformStamped> tfs{
                static_tf(now, frames.imu, frames.lidar, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU)};
            if (!frames.map.empty() && !loc::state.enabled)   // localization: map -> init once localized
                tfs.push_back(static_tf(now, frames.map, frames.init, Eye3d, Zero3d));
            tf_static_broadcaster_->sendTransform(tfs);
        }
        if (publish_registered_cloud)
            pubRegisteredCloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(registered_cloud_topic, 20);
        if (publish_map)
            pubMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, 1);
        if (loc::state.enabled)
            start_localization_io();

        RCLCPP_INFO(this->get_logger(), "Node init finished.");
    }

public:
    ~LaserMappingNode() { stop_processing(); }

    // Live: the processing thread, woken by each sensor message.
    void start_processing()
    {
        processing_thread_ = std::thread([this] {
            while (true)
            {
                {
                    std::unique_lock<std::mutex> lock(mtx_buffer);
                    sig_buffer.wait(lock, [this] { return new_input || stop_; });
                    if (stop_)
                        return;
                    new_input = false;
                }
                while (process_next_scan())
                    ;
            }
        });
    }

    void stop_processing()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            stop_ = true;
        }
        sig_buffer.notify_one();
        if (processing_thread_.joinable())
            processing_thread_.join();
    }

    // Processes the next scan once IMU data past its end has arrived (sync_packages). Returns false if
    // there was nothing to process. Called by the processing thread (live), or by the offline replay.
    bool process_next_scan()
    {
        {
            std::lock_guard<std::mutex> lock(mtx_buffer);
            if (!sync_packages(Measures))
                return false;
        }
        dbg::session.scan_index++;
        if (flg_first_scan)
        {
            first_lidar_time = Measures.lidar_beg_time;
            flg_first_scan = false;
            p_imu->discard_until(Measures.lidar_end_time);   // the first scan is not processed
            return true;
        }
        process_scan();
        // the state at the scan end becomes the base; the newer IMU samples are integrated again from it. A jump
        // of this scan goes into init -> odom at the same moment, so the IMU thread's odom -> imu stays continuous.
        {
            std::lock_guard<std::mutex> lock(mtx_frames);
            p_imu->end_scan(kf);
            if (!jump_.identity())
                init_odom = jump_ * init_odom;
        }
        jump_ = Rigid();
        return true;
    }

private:
    // One scan: the filter is rolled back to the scan end, the scan deskewed, matched and added to the map.
    void process_scan()
    {
        double t0, t1, t3, t5;

        match_time = 0;
        solve_time = 0;
        t0 = omp_get_wtime();
        jump_ = Rigid();

        if (p_imu->begin_scan(*Measures.lidar, Measures.lidar_beg_time, Measures.lidar_end_time, kf,
                              *feats_undistort) != ImuProcess::DESKEWED)
            return;
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
            return;
        }

        flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                        false : true;
        /*** Segment the map in lidar FOV (ikd-Tree only) ***/
        if (use_ikdtree)
            lasermap_fov_segment();

        /*** downsample the feature points in a scan ***/
        downsample_scan();
        t1 = omp_get_wtime();
        /*** initialize the maps with the first scan ***/
        if (!use_ikdtree && voxel_map.empty() && !loc::state.active())
        {
            RCLCPP_INFO(this->get_logger(), "Initialize the voxel map");
            voxel_map_incremental();
            return;
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
            return;
        }

        /*** ICP and iterated Kalman filter update ***/
        if (feats_down_size < 5)
        {
            RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
            return;
        }

        resize_per_point_arrays();
        update_calls = 0;

        /*** debug session: stop here at the requested scan, then carry on with the real update ***/
        if (dbg::session.active() && !dbg::session.quit && dbg::session.scan_index >= dbg::session.target_scan &&
            !dbg::session.serve(Measures.lidar_beg_time, lidar_end_time))
            return;

        /*** iterated state estimation ***/
        double t_update_start = omp_get_wtime();
        double solve_H_time = 0;
        const state_ikfom x_prior = state_point;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        if (dbg::session.active())
            dbg::session.add_pose(lidar_end_time, state_point);

        double t_update_end = omp_get_wtime();

        /*** a large correction is a jump (init -> odom); localization ***/
        const Rigid prior_pose{x_prior.rot.toRotationMatrix(), x_prior.pos};
        const Rigid post_pose{state_point.rot.toRotationMatrix(), state_point.pos};
        const Rigid correction = post_pose * prior_pose.inverse();
        if ((state_point.pos - x_prior.pos).norm() > jump_distance ||
            Eigen::AngleAxisd(correction.R).angle() > jump_angle)
        {
            jump_ = correction;
            RCLCPP_INFO(this->get_logger(), "Jump at %.3f: the update moved the pose by %.3f m, %.2f deg (into %s -> %s)",
                        lidar_end_time, (state_point.pos - x_prior.pos).norm(),
                        Eigen::AngleAxisd(correction.R).angle() * 180 / M_PI, frames.init.c_str(), frames.odom.c_str());
        }
        if (loc::state.enabled)
            localization_step();

        /******* Publish odometry *******/
        const Rigid io = jump_.identity() ? init_odom : jump_ * init_odom;   // init -> odom after this scan
        if (!loc::state.enabled || loc::state.localized)   // the replay's TUM output: init (SLAM) or map
        {
            state_ikfom x = state_point;
            Filter::cov P = kf.get_P();
            move_state(loc::state.map_init, x, P);
            fill_odometry(odomAftMapped, lidar_end_time, x, P, loc::state.enabled ? frames.map : frames.init);
            odom_count++;
        }
        if (odometry_imu_every <= 0)
            publish_odometry(lidar_end_time, state_point, kf.get_P(), p_imu->scan_angvel(), io);

        /*** add the feature points to map kdtree ***/
        t3 = omp_get_wtime();
        map_incremental();
        t5 = omp_get_wtime();

        /******* Publish points *******/
        Rigid T_out;
        const string *out_frame = nullptr;
        if (output_frame(io, T_out, out_frame))
        {
            if (path_en)                     publish_path(pubPath_, T_out, *out_frame);
            if (scan_pub_en)                 publish_frame_world(pubLaserCloudFull_, T_out, *out_frame);
            if (publish_registered_cloud)   // the full deskewed scan
                publish_xyzi(pubRegisteredCloud_, *feats_undistort,
                             [&T_out](const V3D &p) { return T_out * lidar_to_world(p); }, out_frame->c_str());
            if (publish_map)                 publish_map_cloud(pubMap_, T_out, *out_frame);
        }
        if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);

        /*** Timing log ***/
        if (runtime_pos_log)
        {
            frame_num ++;
            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
            aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
            aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
            aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
            aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
            TimeLogRow row{Measures.lidar_beg_time, t5 - t0, current_preprocess_time,
                           int(feats_undistort->points.size()), feats_down_size,
                           int(use_ikdtree ? ikdtree.size() : voxel_map.size()), matched_last, update_calls,
                           res_mean_last};
            {
                const V3D g = state_point.grav;
                row.grav_tilt = acos(std::min(1.0, -g(2) / g.norm())) * 180 / M_PI;
                const M3D R = state_point.rot.toRotationMatrix();
                row.roll = atan2(R(2, 1), R(2, 2)) * 180 / M_PI;
                row.pitch = -asin(std::max(-1.0, std::min(1.0, R(2, 0)))) * 180 / M_PI;
                const Eigen::AngleAxisd aa(state_point.offset_R_L_I.toRotationMatrix());
                for (int k = 0; k < 3; k++)
                {
                    row.ext_rot[k] = aa.angle() * aa.axis()(k) * 180 / M_PI;
                    row.no_match = no_match_last;
                    row.gate_fail = gate_fail_last;
                    row.pooled = pooled_last;
                    row.blob = blob_last;
                    row.ba[k] = state_point.ba(k);
                    row.bg[k] = state_point.bg(k);
                    row.lam_t[k] = lam_t_last(k);
                    row.lam_r[k] = lam_r_last(k);
                    row.weak_t[k] = weak_t_last(k);
                }
                row.free_space_time = free_space_time;
                row.removed = removed_last;
            }
            time_log.push_back(row);
            printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
        }
    }

    // The odometry at every odometry_imu_every-th IMU sample (IMU thread): the state at the newest sample.
    void publish_imu_odometry()
    {
        double t;
        state_ikfom x;
        Filter::cov P;
        V3D angvel;
        std::lock_guard<std::mutex> lock(mtx_frames);   // the state with init -> odom and map -> init of its time
        if (!p_imu->newest_state(t, x, P, angvel))
            return;
        if (imu_odom_skip_-- > 0)
            return;
        imu_odom_skip_ = odometry_imu_every - 1;
        publish_odometry(t, x, P, angvel, init_odom);
    }

    // The odometry (in the output frame) and TF init -> odom -> imu, from one thread: the IMU thread
    // (odometry_imu_every > 0) or the processing thread (at the scan end); x in init, io: init -> odom. The first
    // one also fixes the lidar_level frame.
    void publish_odometry(double t, const state_ikfom &x_init, const Filter::cov &P_init, const V3D &angvel,
                          const Rigid &io)
    {
        if (!level_known_)
        {
            R_I_level_ = p_imu->initial_rotation().transpose();
            level_known_ = true;
            if (publish_tf)
                tf_static_broadcaster_->sendTransform(
                    static_tf(get_ros_time(t), frames.imu, frames.lidar_level, R_I_level_, Lidar_T_wrt_IMU));
        }
        Rigid T;
        const string *parent = nullptr;
        if (output_frame(io, T, parent))
        {
            state_ikfom x = x_init;
            Filter::cov P = P_init;
            move_state(T, x, P);
            nav_msgs::msg::Odometry odom;
            switch (odometry_frame)
            {
            case OdomFrame::IMU:
                fill_odometry(odom, t, x, P, *parent);
                fill_twist(odom, x, P, angvel, Eye3d, Zero3d);
                break;
            case OdomFrame::LIDAR:
                fill_frame_odometry(odom, t, x, P, angvel, Lidar_R_wrt_IMU, Lidar_T_wrt_IMU, frames.lidar, *parent);
                break;
            case OdomFrame::LIDAR_LEVEL:
                fill_frame_odometry(odom, t, x, P, angvel, R_I_level_, Lidar_T_wrt_IMU, frames.lidar_level, *parent);
                break;
            }
            pubOdom_->publish(odom);
        }
        if (publish_tf)
        {
            state_ikfom x = x_init;   // odom -> imu
            Filter::cov P = Filter::cov::Zero();
            move_state(io.inverse(), x, P);
            geometry_msgs::msg::TransformStamped trans;
            trans.header.stamp = get_ros_time(t);
            trans.header.frame_id = frames.odom;
            trans.child_frame_id = frames.imu;
            trans.transform.translation.x = x.pos(0);
            trans.transform.translation.y = x.pos(1);
            trans.transform.translation.z = x.pos(2);
            trans.transform.rotation.x = x.rot.coeffs()[0];
            trans.transform.rotation.y = x.rot.coeffs()[1];
            trans.transform.rotation.z = x.rot.coeffs()[2];
            trans.transform.rotation.w = x.rot.coeffs()[3];
            tf_broadcaster_->sendTransform({static_tf(trans.header.stamp, frames.init, frames.odom, io.R, io.t), trans});
        }
    }

    // A TF parent -> child (R: child to parent, t: the child's origin in the parent frame).
    static geometry_msgs::msg::TransformStamped static_tf(const builtin_interfaces::msg::Time &stamp,
                                                          const string &parent, const string &child,
                                                          const M3D &R, const V3D &t)
    {
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = stamp;
        tf.header.frame_id = parent;
        tf.child_frame_id = child;
        tf.transform.translation.x = t(0);
        tf.transform.translation.y = t(1);
        tf.transform.translation.z = t(2);
        const Eigen::Quaterniond q(R);
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf.transform.rotation.w = q.w();
        return tf;
    }

    // localization.*: the primary map and the relocalization (localization.hpp).
    void setup_localization()
    {
        string map_file;
        this->get_parameter_or<string>("localization.map_file", map_file, "");
        map_file = expand_home(map_file);
        if (map_file.empty())
            return;
        const auto stop = [this, &map_file](const string &why) {
            RCLCPP_FATAL(this->get_logger(), "Localization map %s: %s", map_file.c_str(), why.c_str());
            throw std::invalid_argument("localization.map_file");
        };
        if (use_ikdtree)
            stop(VoxelMap::is_vmap(map_file) ? "a voxel map, but map_type is ikdtree"
                                             : "localization on an ikd-Tree map is not implemented yet");
        if (frames.map.empty())
            stop("localization needs a frames.map name");
        string error;
        Eigen::Isometry3d frame;
        if (!loc::state.prior.load(map_file, &error, &frame))
            stop(error);
        const string differ = loc::state.prior.params().differences(voxel_map.params());
        if (!differ.empty())
            RCLCPP_WARN(this->get_logger(), "The primary map's voxel parameters differ from voxel_map.*, which apply to the "
                        "live layer only (map / config): %s", differ.c_str());
        if (!frame.isApprox(Eigen::Isometry3d::Identity()))
            RCLCPP_WARN(this->get_logger(), "%s is a live layer saved in localization mode: its own frame is the map frame "
                        "here", map_file.c_str());
        loc::state.enabled = true;
        this->get_parameter("localization.submap_scans", loc::state.submap_scans);
        this->get_parameter("localization.live_radius", loc::state.live_radius);
        Relocalizer::Params &rp = loc::state.params;
        this->get_parameter("localization.search_xy", rp.xy_window);
        this->get_parameter("localization.search_yaw_deg", rp.yaw_window);
        this->get_parameter("localization.search_z", rp.z_window);
        this->get_parameter("localization.max_range", rp.max_range);
        vector<double> band;
        this->get_parameter("localization.score_band", band);
        if (band.size() == 2)
            rp.band_low = band[0], rp.band_high = band[1];
        else
            RCLCPP_WARN(this->get_logger(), "localization.score_band needs [low, high]; using [%.2f, %.2f]",
                        rp.band_low, rp.band_high);
        this->get_parameter("localization.accept_matched", rp.accept_matched);
        this->get_parameter("localization.max_ambiguity", rp.max_ambiguity);
        loc::state.reloc = std::make_unique<Relocalizer>(loc::state.prior, rp);
        vector<double> pose;
        this->get_parameter_or<vector<double>>("localization.initial_pose", pose, vector<double>());
        if (pose.size() == 3)
            loc::set_guess(pose[0], pose[1], pose[2] * M_PI / 180);
        else
        {
            if (!pose.empty())
                RCLCPP_WARN(this->get_logger(), "localization.initial_pose needs [x, y, yaw_deg]; searching the whole map");
            loc::state.global_pending = true;   // no guess: the search over the whole map, once
        }
        if (loc::state.reloc->has_floor())
            RCLCPP_INFO(this->get_logger(), "Localization on %s: %zu voxels of %.2f m, floor at z %.2f m", map_file.c_str(),
                        loc::state.prior.size(), loc::state.prior.params().size, loc::state.reloc->floor_z());
        else
            RCLCPP_WARN(this->get_logger(), "Localization on %s: %zu voxels of %.2f m, no floor found: relocalization "
                        "assumes the mapping sensor's height", map_file.c_str(), loc::state.prior.size(),
                        loc::state.prior.params().size);
    }

    // Localization topics: the primary map (latched, points as map_topic's: publish.map_point_step apart, colored by
    // the plane normal), the status (latched) and the initial pose (e.g. Foxglove's or RViz's pose estimate tool, in
    // the map frame).
    void start_localization_io()
    {
        string map_topic, pose_topic, status_topic;
        this->get_parameter_or<string>("localization.map_topic", map_topic, "/prior_map");
        this->get_parameter_or<string>("localization.initial_pose_topic", pose_topic, "/initialpose");
        this->get_parameter_or<string>("localization.status_topic", status_topic, "/localization_status");
        const auto latched = rclcpp::QoS(1).transient_local();
        pubStatus_ = this->create_publisher<std_msgs::msg::String>(status_topic, latched);
        pubPriorMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(map_topic, latched);
        PointCloudXYZI cloud;
        loc::state.prior.to_cloud(cloud, map_point_step);
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = frames.map;
        map_cloud_msg(cloud, Rigid(), true, msg);
        pubPriorMap_->publish(msg);
        sub_initial_pose_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(pose_topic, 1,
            [this](geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr m) {
                if (m->header.frame_id != frames.map)
                {
                    RCLCPP_WARN(this->get_logger(), "Initial pose ignored: frame '%s', expected '%s'",
                                m->header.frame_id.c_str(), frames.map.c_str());
                    return;
                }
                const auto &q = m->pose.pose.orientation;
                const double yaw = std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
                loc::set_guess(m->pose.pose.position.x, m->pose.pose.position.y, yaw);
                set_status("initial pose received");
            });
        set_status(loc::state.has_guess ? "waiting for scans" : "waiting for scans (search over the whole map)");
    }

    // After the update (processing thread): the scan goes to the submap, a relocalization may start or finish,
    // and the live layer is kept within its radius.
    void localization_step()
    {
        const int started = loc::after_scan(state_point, *feats_undistort);
        if (started)
            set_status(started == 2 ? "searching the whole map" : "relocalizing");
        loc::Job job;
        if (loc::finished(job))
        {
            const Relocalizer::Result &r = job.res;
            char buf[200];
            if (!r.ok)
                snprintf(buf, sizeof(buf), "%s failed (%s): %.0f %% of the points matched, next best %.0f %%%s",
                         job.global ? "the search over the whole map" : "relocalization", r.reason.c_str(), 100 * r.matched,
                         100 * r.second, loc::state.localized ? "" : "; waiting for an initial pose");
            else
            {
                const bool first = !loc::state.localized;
                Rigid D;
                {
                    std::lock_guard<std::mutex> lock(mtx_frames);   // map -> init, read by the IMU thread
                    D = loc::apply(job);
                }
                if (!first)
                {
                    // map -> init stays: the state moves, and the move is a jump
                    state_ikfom x = kf.get_x();
                    Filter::cov P = kf.get_P();
                    move_state(D, x, P);
                    kf.change_x(x);
                    kf.change_P(P);
                    state_point = x;
                    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
                    jump_ = D * jump_;
                }
                else if (publish_tf)
                    tf_static_broadcaster_->sendTransform(static_tf(get_ros_time(lidar_end_time), frames.map, frames.init,
                                                                    loc::state.map_init.R, loc::state.map_init.t));
                snprintf(buf, sizeof(buf), "localized%s at x %.2f, y %.2f, yaw %.1f deg: %.0f %% of the points matched, "
                         "next best %.0f %% (%.0f ms)", job.global ? " (search over the whole map)" : "", r.pose.x, r.pose.y,
                         r.pose.yaw * 180 / M_PI, 100 * r.matched, 100 * r.second, r.search_ms + r.refine_ms);
            }
            set_status(buf);
        }
        if (loc::state.live_radius > 0 && ++live_trim_count_ % loc::TRIM_EVERY == 0)
            voxel_map.trim(state_point.pos, loc::state.live_radius);
    }

    void set_status(const string &status)
    {
        std::lock_guard<std::mutex> lock(mtx_status_);
        if (status == status_)
            return;
        status_ = status;
        RCLCPP_INFO(this->get_logger(), "Localization: %s", status.c_str());
        std_msgs::msg::String msg;
        msg.data = status;
        pubStatus_->publish(msg);
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdom_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;   // odom -> imu, with the odometry
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRegisteredCloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubMap_;
    // Localization mode (localization.hpp):
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pubStatus_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorMap_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_initial_pose_;
    std::mutex mtx_status_;
    string status_;
    int live_trim_count_ = 0;
    Rigid jump_;                        // this scan's jump (processing thread)
    // The odometry's thread only (publish_odometry):
    int  imu_odom_skip_ = 0;            // IMU samples to skip before the next odometry
    bool level_known_ = false;
    M3D  R_I_level_ = Eye3d;            // the lidar_level frame: lidar_level to IMU rotation
    std::thread processing_thread_;
    bool stop_ = false;        // guarded by mtx_buffer

    int frame_num = 0;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    double epsi[23] = {0.001};
};

// The timing log (runtime_pos_log_enable) as CSV: time_stamp [s], total time [s] (scan processing, from
// deskewing to the map update), preprocess time [s], scan point size, down points, map size, matched,
// update calls, residual mean [m], ..., free-space time [s], voxels removed.
void write_time_log()
{
    const string path = time_log_path.empty() ? "fast_lio_time_log.csv" : time_log_path;
    FILE *fp = fopen(path.c_str(), "w");
    if (fp == nullptr)
    {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    fprintf(fp, "time_stamp, total time, preprocess time, scan point size, down points, map size, matched, "
                "update calls, residual mean, grav_tilt_deg, roll_deg, pitch_deg, ba_x, ba_y, ba_z, bg_x, bg_y, bg_z, "
                "lam_t0, lam_t1, lam_t2, lam_r0, lam_r1, lam_r2, weak_tx, weak_ty, weak_tz, ext_rx, ext_ry, ext_rz, "
                "no_match, gate_fail, pooled, blob, free_space_time, removed\n");
    for (const auto &r : time_log)
        fprintf(fp, "%0.8f,%0.8f,%0.8f,%d,%d,%d,%d,%d,%0.6f,%0.4f,%0.4f,%0.4f,%0.5f,%0.5f,%0.5f,%0.6f,%0.6f,%0.6f,"
                    "%0.1f,%0.1f,%0.1f,%0.1f,%0.1f,%0.1f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%0.4f,%d,%d,%d,%d,%0.6f,%d\n",
                r.stamp, r.total, r.preprocess, r.scan_points, r.down_points, r.map_size, r.matched, r.update_calls,
                r.residual_mean, r.grav_tilt, r.roll, r.pitch, r.ba[0], r.ba[1], r.ba[2], r.bg[0], r.bg[1], r.bg[2],
                r.lam_t[0], r.lam_t[1], r.lam_t[2], r.lam_r[0], r.lam_r[1], r.lam_r[2], r.weak_t[0], r.weak_t[1],
                r.weak_t[2], r.ext_rot[0], r.ext_rot[1], r.ext_rot[2], r.no_match, r.gate_fail,
                r.pooled, r.blob, r.free_space_time, r.removed);
    fclose(fp);
}

// Live:    fastlio_mapping --ros-args --params-file <config> ...
// Offline: fastlio_mapping --replay <dataset.g1replay> <odom.tum> [--map <name>] [--debug-port <port>]
//          [--dump-scans <file>] --ros-args ...
//          feeds a recorded dataset straight into the callbacks (no DDS), writes the odometry as TUM and exits
//          (replay.hpp). --map writes the final map (save_map: <name>.vmap or .pcd by map_type); --debug-port serves the
//          debug session (debug_session.hpp); --dump-scans writes the scans as they went into the voxel map
//          (scan_dump).
// Live with map_save: at shutdown (SIGINT) the node leaves ROS first, then the map is written (save_map), so a new
// stack can start meanwhile.
int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    const vector<string> args = rclcpp::remove_ros_arguments(argc, argv);
    auto node = std::make_shared<LaserMappingNode>();
    int ret = 0;
    bool save_at_exit = false;
    if (args.size() > 1 && args[1] == "--replay")
    {
#ifndef FASTLIO_REPLAY
        fprintf(stderr, "fastlio_mapping was built without zlib: no --replay\n");
        return 2;
#else
        string map_out;
        int debug_port = 0;
        bool usage_ok = args.size() >= 4;
        for (size_t i = 4; usage_ok && i < args.size(); i += 2)
        {
            if (i + 1 >= args.size())
                usage_ok = false;
            else if (args[i] == "--map")
                map_out = args[i + 1];
            else if (args[i] == "--debug-port")
                debug_port = atoi(args[i + 1].c_str());
            else if (args[i] == "--dump-scans")
            {
                scan_dump = fopen(args[i + 1].c_str(), "wb");
                if (scan_dump == nullptr)
                {
                    fprintf(stderr, "cannot write %s\n", args[i + 1].c_str());
                    return 2;
                }
            }
            else
                usage_ok = false;
        }
        if (!usage_ok)
        {
            fprintf(stderr, "usage: fastlio_mapping --replay <dataset.g1replay> <odom.tum> [--map <name>] [--debug-port <port>] "
                            "[--dump-scans <file>] [--ros-args ...]\n");
            return 2;
        }
        if (debug_port > 0)
        {
            dbg::session.port = debug_port;
            dbg::session.start();
        }
        // The replay feeds each scan before the IMU that completes the previous one and processes every
        // scan in order (lockstep): the live limits (1 scan, 1 s of IMU) would drop data here.
        max_lidar_buffer = 100;
        p_imu->set_max_samples(2000);
        loc::state.synchronous = true;   // a relocalization finishes within its scan: the replay is repeatable
        uint64_t poses_taken = 0;
        replay::Hooks hooks;
        hooks.imu = [](sensor_msgs::msg::Imu::UniquePtr m) { imu_cbk(std::move(m)); };
        hooks.lidar = [](sensor_msgs::msg::PointCloud2::UniquePtr m) { standard_pcl_cbk(std::move(m)); };
        hooks.process = [&node]() { return node->process_next_scan(); };
        hooks.ok = []() { return rclcpp::ok() && !dbg::session.quit; };
        hooks.new_pose = [&poses_taken](nav_msgs::msg::Odometry &odom) {
            if (odom_count == poses_taken)
                return false;
            poses_taken = odom_count;
            odom = odomAftMapped;
            return true;
        };
        ret = replay::run(args[2], args[3], hooks);
        if (imu_restamp.restamper)
            imu_restamp.log();
        if (scan_dump != nullptr)
            fclose(scan_dump);
        if (debug_port > 0)
        {
            dbg::session.finish();
            if (ret == 130 && dbg::session.quit)
                ret = 0;   // the viewer ended the session
        }
        if (!map_out.empty())
        {
            map_file_path = map_out;
            double leveled_deg = 0;
            const string path = save_map(&leveled_deg);
            if (path.empty())
                fprintf(stderr, "map not saved: no map, or %s cannot be written\n", map_file_path.c_str());
            else
                fprintf(stderr, "map written to %s (leveled by %.2f deg)\n", path.c_str(), leveled_deg);
        }
#endif
    }
    else
    {
        node->start_processing();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);   // IMU, lidar
        executor.add_node(node);
        executor.spin();
        node->stop_processing();
        save_at_exit = map_save_en;
    }
    node.reset();
    if (rclcpp::ok())
        rclcpp::shutdown();
    if (save_at_exit)
    {
        double leveled_deg = 0;
        const string path = save_map(&leveled_deg);
        if (path.empty())
            fprintf(stderr, "map not saved at exit: no map yet, or %s cannot be written\n", map_file_path.c_str());
        else
            fprintf(stderr, "map saved at exit to %s (leveled by %.2f deg)\n", path.c_str(), leveled_deg);
    }
    if (runtime_pos_log)
        write_time_log();
    return ret;
}

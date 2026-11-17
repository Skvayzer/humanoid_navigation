#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/imu.hpp>

using namespace std;
using namespace Eigen;

#define G_m_s2 (9.81)         // gravity [m/s^2]; also the S2 length in use-ikfom.hpp (98100 / 10000)
#define NUM_MATCH_POINTS    (5)

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]

// IMU-propagated state at one IMU sample inside a scan (all in the world frame except gyr).
struct Pose6D
{
    double offset_time;  // time of the IMU sample w.r.t. the scan start [s]
    double acc[3];       // acceleration, world frame, gravity removed
    double gyr[3];       // unbiased angular velocity, body frame
    double vel[3];
    double pos[3];
    double rot[9];       // row-major rotation matrix
};

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>>  PointVector;
typedef Vector3d V3D;
typedef Matrix3d M3D;

#define VF(a)    Matrix<float, (a), 1>

M3D Eye3d(M3D::Identity());
V3D Zero3d(0, 0, 0);

struct MeasureGroup     // the scan being processed (its IMU samples are in ImuProcess)
{
    MeasureGroup()
    {
        lidar_beg_time = 0.0;
        this->lidar.reset(new PointCloudXYZI());
    };
    double lidar_beg_time;
    double lidar_end_time;
    PointCloudXYZI::Ptr lidar;
};

template<typename T>
Pose6D set_pose6d(const double t, const Matrix<T, 3, 1> &a, const Matrix<T, 3, 1> &g,
                  const Matrix<T, 3, 1> &v, const Matrix<T, 3, 1> &p, const Matrix<T, 3, 3> &R)
{
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++)
    {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++)  rot_kp.rot[i*3+j] = R(i,j);
    }
    return rot_kp;
}

inline float calc_dist(const PointType &p1, const PointType &p2){
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

// Inverse of the Cholesky factor of a symmetric positive-definite 3x3 matrix C = L L', in closed form
// (Eigen's LLT and eigen solver are generic and several times slower for 3x3). M = L^-1 is lower
// triangular, and M' M = C^-1: |M r|^2 = r' C^-1 r, and the rows of M whiten a 3-D residual.
inline M3D inverse_cholesky3(const M3D &C)
{
    const double l00 = std::sqrt(C(0, 0));
    const double l10 = C(1, 0) / l00, l20 = C(2, 0) / l00;
    const double l11 = std::sqrt(C(1, 1) - l10 * l10);
    const double l21 = (C(2, 1) - l20 * l10) / l11;
    const double l22 = std::sqrt(C(2, 2) - l20 * l20 - l21 * l21);
    M3D M = M3D::Zero();
    M(0, 0) = 1.0 / l00;
    M(1, 1) = 1.0 / l11;
    M(2, 2) = 1.0 / l22;
    M(1, 0) = -l10 * M(0, 0) * M(1, 1);
    M(2, 1) = -l21 * M(1, 1) * M(2, 2);
    M(2, 0) = -(l20 * M(0, 0) + l21 * M(1, 0)) * M(2, 2);
    return M;
}

template<typename T>
bool esti_plane(Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Matrix<T, NUM_MATCH_POINTS, 3> A;
    Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
        {
            return false;
        }
    }
    return true;
}

double get_time_sec(const builtin_interfaces::msg::Time &time)
{
    return rclcpp::Time(time).seconds();
}

// Seconds -> stamp, rounded to the nearest nanosecond.
rclcpp::Time get_ros_time(double timestamp)
{
    int32_t sec = std::floor(timestamp);
    int64_t nanosec = llround((timestamp - sec) * 1e9);
    if (nanosec >= 1000000000)
    {
        sec++;
        nanosec -= 1000000000;
    }
    return rclcpp::Time(sec, uint32_t(nanosec));
}

#endif
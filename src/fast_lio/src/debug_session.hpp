#pragma once
// Interactive debug session for the map and matching design:
//   fastlio_mapping --replay <dataset.g1replay> <odom.tum> --debug-port <port> --ros-args ...
// A viewer (scripts/lio_viewer in the project) connects on 127.0.0.1:<port>. The replay runs to a chosen
// scan and stops right before that scan's IEKF update. The IMU-propagated prior, the deskewed scan and the
// map (ikd-Tree points or voxel-map statistics, by map_type) can then be read out, and the update can be run
// any number of times from a perturbed prior, recording every iteration per point (status, the matched voxel
// distribution or the 5-NN plane, residual, the 5 nearest map points), without changing the filter: the state
// and covariance are restored after each run.
// "continue" / "goto" then carry on with the real update as if nothing had happened, so the session can
// step forward through the record.
//
// Protocol: text commands, one per line. Every reply is one JSON line; when it has a "bytes" key, that
// many raw bytes follow (little-endian float32 arrays, strides given in the reply).
//   goto <scan>                  run to that scan (1-based, replay order) -> {"type":"scan", ...}; a scan
//                                without an update (before the IMU is initialised) is skipped to the next
//   map [radius]                 ikd-Tree points and voxel statistics within radius [m] of the lidar
//   scan                         deskewed scan (lidar frame), downsampled points and their patches
//   update dx dy dz rx ry rz     the update from the prior moved by (dx, dy, dz) [m, world frame] and
//                                rotated by Exp(rx, ry, rz) [rad, body frame, the filter's own convention]
//   voxel_flags                  plane flags (voxel_map plane test, used by merging) of the voxels of the
//                                last "map" (after a "set")
//   poses                        the trajectory so far (float64: scan index, stamp, pos, quat xyzw)
//   set <key> <value>            voxel_map.{min_points,max_thickness,planarity,max_points}, scan_patch.min_points,
//                                filter_size_surf, max_iteration, extrinsic_est_en (map_type is fixed at start)
//   continue                     = goto <current scan + 1>
//   quit                         stop the replay
// This header is included by laserMapping.cpp after its globals and works on them directly.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace dbg
{
// Per point and update iteration: 32 float32 (see record_iteration).
constexpr int POINT_STRIDE = 32;
// Per voxel: 32 float32 (see dump_map).
constexpr int VOXEL_STRIDE = 32;
// Per downsampled point: 7 float32 of patch statistics (see dump_scan).
constexpr int PATCH_STRIDE = 7;

inline std::string num(double v)
{
    char b[32];
    snprintf(b, sizeof b, "%.17g", v);
    return b;
}
inline std::string vec(const V3D &v) { return "[" + num(v(0)) + "," + num(v(1)) + "," + num(v(2)) + "]"; }
inline std::string quat(const Eigen::Quaterniond &q)
{
    return "[" + num(q.x()) + "," + num(q.y()) + "," + num(q.z()) + "," + num(q.w()) + "]";
}

struct Session
{
    int port = 0;
    int listen_fd = -1, client_fd = -1;
    bool quit = false;
    int scan_index = 0;      // scans taken from the queue so far = 1-based index of the current scan
    int target_scan = -1;    // stop before the update of this scan (or the next one that has an update)
    bool recording = false;  // h_share_model records its iterations
    struct Iteration
    {
        bool researched;
        V3D pos;
        Eigen::Quaterniond rot;
        int effct, matched, dist, group, counts[5];
        double res_mean;
        std::vector<float> rec;
    };
    std::vector<Iteration> iterations;
    std::vector<double> poses;   // 9 per pose: scan index, stamp, pos (3), quat xyzw
    V3D map_centre = V3D::Zero();
    double map_radius = 0;       // of the last "map" (voxel_flags repeats its selection)
    std::string in_buffer;

    bool active() const { return port > 0; }

    /*** socket ***/
    bool send_all(const void *data, size_t len)
    {
        const char *p = static_cast<const char *>(data);
        while (len > 0)
        {
            const ssize_t n = ::write(client_fd, p, len);
            if (n <= 0)
                return false;
            p += n;
            len -= size_t(n);
        }
        return true;
    }
    // {"fields"[, "bytes": n]}\n + payload
    bool reply(const std::string &fields, const void *payload = nullptr, size_t bytes = 0)
    {
        std::string line = "{" + fields;
        if (payload != nullptr)
            line += ",\"bytes\":" + std::to_string(bytes);
        line += "}\n";
        return send_all(line.data(), line.size()) && (payload == nullptr || send_all(payload, bytes));
    }
    bool reply(const std::string &fields, const std::vector<float> &payload)
    {
        return reply(fields, payload.data(), payload.size() * sizeof(float));
    }
    bool error(const std::string &message) { return reply("\"type\":\"error\",\"message\":\"" + message + "\""); }
    bool read_line(std::string &line)   // false when the client is gone
    {
        while (true)
        {
            const size_t nl = in_buffer.find('\n');
            if (nl != std::string::npos)
            {
                line = in_buffer.substr(0, nl);
                in_buffer.erase(0, nl + 1);
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                return true;
            }
            char buf[4096];
            const ssize_t n = ::read(client_fd, buf, sizeof buf);
            if (n <= 0)
                return false;
            in_buffer.append(buf, size_t(n));
        }
    }

    // Listens, waits for the viewer and for its first "goto". Called before the replay starts.
    void start()
    {
        signal(SIGPIPE, SIG_IGN);
        listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(uint16_t(port));
        if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof addr) != 0 || ::listen(listen_fd, 1) != 0)
        {
            fprintf(stderr, "debug session: cannot listen on 127.0.0.1:%d (%s)\n", port, strerror(errno));
            quit = true;
            return;
        }
        fprintf(stderr, "debug session: waiting for the viewer on 127.0.0.1:%d\n", port);
        client_fd = ::accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0)
        {
            quit = true;
            return;
        }
        reply("\"type\":\"hello\",\"version\":1");
        std::string line;
        while (read_line(line))
        {
            std::istringstream ss(line);
            std::string cmd;
            ss >> cmd;
            if (cmd == "goto" && (ss >> target_scan))
            {
                reply("\"type\":\"running\",\"target\":" + std::to_string(target_scan));
                return;
            }
            if (cmd == "quit")
                break;
            error("the first command must be: goto <scan>");
        }
        quit = true;
    }

    // After the replay: tells the viewer that the record ended (its goto was past the end).
    void finish()
    {
        if (client_fd >= 0)
        {
            reply("\"type\":\"end\",\"scans\":" + std::to_string(scan_index) + ",\"poses\":" + std::to_string(poses.size() / 9));
            ::close(client_fd);
            client_fd = -1;
        }
        if (listen_fd >= 0)
            ::close(listen_fd);
    }

    void add_pose(double stamp, const state_ikfom &s)
    {
        poses.insert(poses.end(), {double(scan_index), stamp, s.pos(0), s.pos(1), s.pos(2), s.rot.x(), s.rot.y(), s.rot.z(), s.rot.w()});
    }

    /*** recording (called by h_share_model at the end of every iteration while recording) ***/
    void record_iteration(const state_ikfom &s, bool researched)
    {
        Iteration it;
        it.researched = researched;
        it.pos = s.pos;
        it.rot = s.rot;
        it.effct = effct_feat_num;
        it.matched = it.dist = it.group = 0;
        std::fill(it.counts, it.counts + 5, 0);
        double res_sum = 0;
        it.rec.assign(size_t(POINT_STRIDE) * feats_down_size, 0.0f);
        for (int i = 0; i < feats_down_size; i++)
        {
            float *r = &it.rec[size_t(POINT_STRIDE) * i];
            // 0 status, 1 matched to a voxel distribution (map_type voxel), 2 neighbours found (0 for a
            // distribution match: the 5-NN list is then stale), 3 residual rows of this point (0: no residual)
            const bool dm = !use_ikdtree && dist_matched[i];
            const int nn = dm ? 0 : int(Nearest_Points[i].size());
            r[0] = point_status[i];
            r[1] = dm;
            r[2] = nn;
            r[3] = point_selected_surf[i] ? row_count[i] : 0;
            // 4-6 the point in the world frame at this iteration's state
            const PointType &pw = feats_down_world->points[i];
            r[4] = pw.x;
            r[5] = pw.y;
            r[6] = pw.z;
            if (dm)
            {
                // 11 distance to the distribution's mean [m], 12-14 the mean (world), 15 a merged group's,
                // 16-21 its covariance xx xy xz yy yz zz, 31 squared Mahalanobis distance (gate D2D_GATE) of
                // the rows set_d2d_rows wrote this iteration (status matched or residual fail)
                const V3D &m = dist_mean[i];
                const M3D &c = dist_cov[i];
                r[11] = (V3D(pw.x, pw.y, pw.z) - m).norm();
                r[12] = m(0);
                r[13] = m(1);
                r[14] = m(2);
                r[15] = from_group[i];
                const float cov[6] = {float(c(0, 0)), float(c(0, 1)), float(c(0, 2)), float(c(1, 1)), float(c(1, 2)), float(c(2, 2))};
                std::copy(cov, cov + 6, r + 16);
                if (point_status[i] == PT_MATCHED || point_status[i] == PT_RESIDUAL_FAIL)
                {
                    double rr = 0;
                    for (int k = 0; k < 3; k++)
                        rr += double(normvec->points[3 * i + k].intensity) * normvec->points[3 * i + k].intensity;
                    r[31] = rr / LASER_POINT_COV;
                }
            }
            else
            {
                // 7-10 plane a b c d (world; valid for matched, residual-fail and, as the failed fit,
                // plane-fail), 11 signed point-to-plane distance
                const VF(4) &pl = plane_coef[i];
                r[7] = pl(0);
                r[8] = pl(1);
                r[9] = pl(2);
                r[10] = pl(3);
                r[11] = pl(0) * pw.x + pl(1) * pw.y + pl(2) * pw.z + pl(3);
            }
            // 16-30 the nearest map points (nn of them)
            for (int k = 0; k < nn && k < NUM_MATCH_POINTS; k++)
            {
                const PointType &q = Nearest_Points[i][k];
                r[16 + 3 * k] = q.x;
                r[17 + 3 * k] = q.y;
                r[18 + 3 * k] = q.z;
            }
            it.counts[point_status[i]]++;
            if (point_selected_surf[i])
            {
                it.matched++;
                it.dist += dm;
                it.group += dm && from_group[i];
                res_sum += res_last[i];
            }
        }
        it.res_mean = it.matched > 0 ? res_sum / it.matched : 0.0;
        iterations.push_back(std::move(it));
    }

    /*** replies ***/
    std::string params_json() const
    {
        const VoxelMap::Params &vp = voxel_map.params();
        return "\"params\":{\"filter_size_surf\":" + num(filter_size_surf_min) + ",\"filter_size_map\":" + num(filter_size_map_min) +
               ",\"max_iteration\":" + std::to_string(NUM_MAX_ITERATIONS) + ",\"extrinsic_est_en\":" + (extrinsic_est_en ? "true" : "false") +
               ",\"map_type\":\"" + (use_ikdtree ? "ikdtree" : "voxel") + "\",\"d2d_gate\":" + num(D2D_GATE) + ",\"patch_var_floor\":" + num(PATCH_VAR_FLOOR) +
               ",\"laser_point_cov\":" + num(LASER_POINT_COV) + ",\"map_insert_rule\":\"" + (map_keep_first_point ? "first_point" : "closest_to_centre") + "\"" +
               ",\"voxel_map\":{\"size\":" + num(vp.size) + ",\"min_points\":" + std::to_string(vp.min_points) + ",\"max_thickness\":" + num(vp.max_thickness) +
               ",\"planarity\":" + num(vp.planarity) + ",\"max_points\":" + std::to_string(vp.max_points) +
               ",\"stored_points\":" + std::to_string(vp.stored_points) + ",\"merge_angle_deg\":" + num(vp.merge_angle_deg) +
               ",\"merge_distance\":" + num(vp.merge_distance) + "}" +
               ",\"scan_patch\":{\"min_points\":" + std::to_string(scan_patch_min_points) + "}}";
    }

    std::string scan_header(double stamp, double end_stamp) const
    {
        const state_ikfom &s = state_point;   // the prior: kf.get_x() after the IMU propagation
        const auto &P = kf.get_P();
        std::string p6 = "[";
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                p6 += (i + j > 0 ? "," : "") + num(P(i, j));
        p6 += "]";
        return "\"type\":\"scan\",\"index\":" + std::to_string(scan_index) + ",\"stamp\":" + num(stamp) + ",\"end_stamp\":" + num(end_stamp) +
               ",\"raw_points\":" + std::to_string(feats_undistort->points.size()) + ",\"down_points\":" + std::to_string(feats_down_size) +
               ",\"patches\":" + std::to_string(scan_patch_min_points > 0 ? scan_patch_count : 0) +
               ",\"prior\":{\"pos\":" + vec(s.pos) + ",\"quat\":" + quat(s.rot) + ",\"vel\":" + vec(s.vel) + ",\"bg\":" + vec(s.bg) + ",\"ba\":" + vec(s.ba) +
               ",\"grav\":[" + num(s.grav[0]) + "," + num(s.grav[1]) + "," + num(s.grav[2]) + "]}" +
               ",\"extrinsic\":{\"quat\":" + quat(s.offset_R_L_I) + ",\"t\":" + vec(s.offset_T_L_I) + "},\"lidar_pos\":" + vec(pos_lid) +
               ",\"P6\":" + p6 + "," + params_json() + ",\"ikd_size\":" + std::to_string(use_ikdtree && ikdtree.Root_Node != nullptr ? ikdtree.validnum() : 0) +
               ",\"voxels\":" + std::to_string(voxel_map.size()) + ",\"poses\":" + std::to_string(poses.size() / 9);
    }

    // ikd-Tree points (4 float32: x y z intensity) and voxels (VOXEL_STRIDE float32: index (3), n, w, mean (3),
    // covariance xx xy xz yy yz zz, normal (3), d, eigenvalues ascending (3), eigenvectors (3 x 3, the first
    // is the normal direction), valid, frozen) within `radius` of the lidar.
    bool dump_map(double radius)
    {
        map_centre = pos_lid;
        map_radius = radius;
        std::vector<float> out;
        size_t n_ikd = 0;
        if (use_ikdtree && ikdtree.Root_Node != nullptr)
        {
            PointVector pts;
            ikdtree.flatten(ikdtree.Root_Node, pts, NOT_RECORD);
            out.reserve(4 * pts.size());
            for (const PointType &p : pts)
            {
                if ((V3D(p.x, p.y, p.z) - map_centre).norm() > radius)
                    continue;
                out.insert(out.end(), {p.x, p.y, p.z, p.intensity});
                n_ikd++;
            }
        }
        size_t n_vox = 0;
        voxel_map.for_each([&](const VoxelMap::Info &v) {
            if ((v.mean - map_centre).norm() > radius)
                return;
            Eigen::SelfAdjointEigenSolver<M3D> es;
            es.computeDirect(v.cov);
            const V3D l = es.eigenvalues();
            const M3D e = es.eigenvectors();
            const float rec[VOXEL_STRIDE] = {float(v.index(0)), float(v.index(1)), float(v.index(2)), float(v.n), float(v.w),
                                             float(v.mean(0)), float(v.mean(1)), float(v.mean(2)),
                                             float(v.cov(0, 0)), float(v.cov(0, 1)), float(v.cov(0, 2)), float(v.cov(1, 1)), float(v.cov(1, 2)), float(v.cov(2, 2)),
                                             float(v.normal(0)), float(v.normal(1)), float(v.normal(2)), float(v.d),
                                             float(l(0)), float(l(1)), float(l(2)),
                                             float(e(0, 0)), float(e(1, 0)), float(e(2, 0)), float(e(0, 1)), float(e(1, 1)), float(e(2, 1)),
                                             float(e(0, 2)), float(e(1, 2)), float(e(2, 2)), float(v.valid), float(v.frozen)};
            out.insert(out.end(), rec, rec + VOXEL_STRIDE);
            n_vox++;
        });
        return reply("\"type\":\"map\",\"ikd_points\":" + std::to_string(n_ikd) + ",\"voxels\":" + std::to_string(n_vox) + ",\"radius\":" + num(radius) +
                     ",\"centre\":" + vec(map_centre) + ",\"ikd_stride\":4,\"voxel_stride\":" + std::to_string(VOXEL_STRIDE) + "," + params_json(), out);
    }

    bool dump_voxel_flags()
    {
        std::vector<float> out;
        voxel_map.for_each([&](const VoxelMap::Info &v) {
            if ((v.mean - map_centre).norm() > map_radius)
                return;
            out.push_back(float(v.valid));
        });
        return reply("\"type\":\"voxel_flags\",\"voxels\":" + std::to_string(out.size()), out);
    }

    // The deskewed scan (lidar frame, 4 float32 per point), the downsampled points (4 float32) and their patch
    // statistics (PATCH_STRIDE float32: points, covariance xx xy xz yy yz zz; lidar frame).
    bool dump_scan()
    {
        std::vector<float> out;
        out.reserve(4 * feats_undistort->points.size() + (4 + PATCH_STRIDE) * feats_down_size);
        for (const PointType &p : feats_undistort->points)
            out.insert(out.end(), {p.x, p.y, p.z, p.intensity});
        for (int i = 0; i < feats_down_size; i++)
        {
            const PointType &p = feats_down_body->points[i];
            out.insert(out.end(), {p.x, p.y, p.z, p.intensity});
        }
        for (int i = 0; i < feats_down_size; i++)
        {
            const ScanPatch &sp = i < int(scan_patches.size()) ? scan_patches[i] : ScanPatch();
            const float rec[PATCH_STRIDE] = {float(sp.n), float(sp.cov(0, 0)), float(sp.cov(0, 1)), float(sp.cov(0, 2)),
                                             float(sp.cov(1, 1)), float(sp.cov(1, 2)), float(sp.cov(2, 2))};
            out.insert(out.end(), rec, rec + PATCH_STRIDE);
        }
        return reply("\"type\":\"scan_data\",\"raw_points\":" + std::to_string(feats_undistort->points.size()) + ",\"down_points\":" + std::to_string(feats_down_size) +
                     ",\"patch_stride\":" + std::to_string(PATCH_STRIDE), out);
    }

    bool dump_poses()
    {
        return reply("\"type\":\"poses\",\"count\":" + std::to_string(poses.size() / 9) + ",\"stride\":9", poses.data(), poses.size() * sizeof(double));
    }

    // The update from the perturbed prior; the filter is restored afterwards.
    bool run_update(const double d[6])
    {
        const state_ikfom prior = kf.get_x();
        const auto P = kf.get_P();
        state_ikfom x = prior;
        x.pos += V3D(d[0], d[1], d[2]);
        x.rot = x.rot * SO3::exp(V3D(d[3], d[4], d[5]));
        kf.change_x(x);
        iterations.clear();
        recording = true;
        double solve_H_time = 0;
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
        recording = false;
        const state_ikfom result = kf.get_x();
        state_ikfom restore = prior;
        auto P_restore = P;
        kf.change_x(restore);
        kf.change_P(P_restore);

        std::string its = "[";
        std::vector<float> out;
        out.reserve(iterations.size() * size_t(POINT_STRIDE) * feats_down_size);
        for (size_t k = 0; k < iterations.size(); k++)
        {
            const Iteration &it = iterations[k];
            its += std::string(k > 0 ? "," : "") + "{\"iter\":" + std::to_string(k) + ",\"researched\":" + (it.researched ? "true" : "false") +
                   ",\"pos\":" + vec(it.pos) + ",\"quat\":" + quat(it.rot) + ",\"effct\":" + std::to_string(it.effct) + ",\"matched\":" + std::to_string(it.matched) +
                   ",\"dist\":" + std::to_string(it.dist) + ",\"group\":" + std::to_string(it.group) + ",\"counts\":[" + std::to_string(it.counts[0]) + "," + std::to_string(it.counts[1]) + "," +
                   std::to_string(it.counts[2]) + "," + std::to_string(it.counts[3]) + "," + std::to_string(it.counts[4]) + "],\"res_mean\":" + num(it.res_mean) + "}";
            out.insert(out.end(), it.rec.begin(), it.rec.end());
        }
        its += "]";
        return reply("\"type\":\"update\",\"perturb\":[" + num(d[0]) + "," + num(d[1]) + "," + num(d[2]) + "," + num(d[3]) + "," + num(d[4]) + "," + num(d[5]) + "]" +
                     ",\"start\":{\"pos\":" + vec(x.pos) + ",\"quat\":" + quat(x.rot) + "},\"final\":{\"pos\":" + vec(result.pos) + ",\"quat\":" + quat(result.rot) + "}" +
                     ",\"iterations\":" + its + ",\"points\":" + std::to_string(feats_down_size) + ",\"stride\":" + std::to_string(POINT_STRIDE), out);
    }

    bool apply_set(const std::string &key, const std::string &value)
    {
        const double v = atof(value.c_str());
        const bool b = v != 0 && value != "false";
        VoxelMap::Params vp = voxel_map.params();
        bool redo_scan = false, refit = false;
        if (key == "voxel_map.min_points") { vp.min_points = int(v); refit = true; }
        else if (key == "voxel_map.max_thickness") { vp.max_thickness = v; refit = true; }
        else if (key == "voxel_map.planarity") { vp.planarity = v; refit = true; }
        else if (key == "voxel_map.max_points") { vp.max_points = int(v); refit = true; }
        else if (key == "scan_patch.min_points") { scan_patch_min_points = int(v); redo_scan = true; }
        else if (key == "filter_size_surf")
        {
            filter_size_surf_min = v;
            downSizeFilterSurf.setLeafSize(v, v, v);
            redo_scan = true;
        }
        else if (key == "max_iteration")
        {
            NUM_MAX_ITERATIONS = int(v);
            kf.set_max_iteration(NUM_MAX_ITERATIONS);
        }
        else if (key == "extrinsic_est_en") extrinsic_est_en = b;
        else
            return error("unknown key " + key);
        int valid = 0;
        if (refit)
        {
            voxel_map.set_params(vp);
            voxel_map.refit_all();
        }
        voxel_map.for_each([&](const VoxelMap::Info &vi) { valid += vi.valid; });
        if (redo_scan)
        {
            downsample_scan();
            resize_per_point_arrays();
            if (scan_patch_min_points <= 0)   // the patches of an earlier setting would stay otherwise
                scan_patches.assign(feats_down_size, ScanPatch());
        }
        return reply("\"type\":\"ok\",\"key\":\"" + key + "\",\"down_points\":" + std::to_string(feats_down_size) + ",\"patches\":" +
                     std::to_string(scan_patch_min_points > 0 ? scan_patch_count : 0) + ",\"voxels_valid\":" + std::to_string(valid) + "," + params_json());
    }

    // At the target scan, right before its update. Returns false to stop the replay (quit or disconnect).
    bool serve(double stamp, double end_stamp)
    {
        if (quit || client_fd < 0)
            return false;
        if (!reply(scan_header(stamp, end_stamp)))
            return quit = true, false;
        std::string line;
        while (read_line(line))
        {
            std::istringstream ss(line);
            std::string cmd;
            ss >> cmd;
            bool ok = true;
            if (cmd == "map")
            {
                double radius = 30.0;
                ss >> radius;
                ok = dump_map(radius);
            }
            else if (cmd == "scan") ok = dump_scan();
            else if (cmd == "voxel_flags") ok = dump_voxel_flags();
            else if (cmd == "poses") ok = dump_poses();
            else if (cmd == "update")
            {
                double d[6] = {0, 0, 0, 0, 0, 0};
                for (double &x : d)
                    ss >> x;
                ok = run_update(d);
            }
            else if (cmd == "set")
            {
                std::string key, value;
                ss >> key >> value;
                ok = apply_set(key, value);
            }
            else if (cmd == "goto")
            {
                int target = 0;
                if (!(ss >> target) || target <= scan_index)
                    ok = error("goto needs a scan after the current one (restart the replay to go back)");
                else
                {
                    target_scan = target;
                    match_time = solve_time = kdtree_search_time = 0;
                    return reply("\"type\":\"running\",\"target\":" + std::to_string(target_scan));
                }
            }
            else if (cmd == "continue")
            {
                target_scan = scan_index + 1;
                match_time = solve_time = kdtree_search_time = 0;
                return reply("\"type\":\"running\",\"target\":" + std::to_string(target_scan));
            }
            else if (cmd == "quit")
                break;
            else if (!cmd.empty())
                ok = error("unknown command " + cmd);
            if (!ok)
                break;
        }
        quit = true;
        return false;
    }
};
Session session;
}  // namespace dbg

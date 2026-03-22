// test_calib_crossval.cpp — Cross-validate laser-refined calibration against
// ChArUco landmarks from the original calibration project.
//
// For each landmark visible in 2+ cameras, triangulates a 3D point using each
// set of camera parameters independently, then measures reprojection error.
// This is a fair comparison: each calibration is evaluated with its own
// best-fit 3D points.
//
// Build: cmake target "test_calib_crossval" (no ImGui/Metal/OpenCV needed).
// Run:   ./test_calib_crossval

#include "json.hpp"
#include "opencv_yaml_io.h"
#include "red_math.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ── Hardcoded paths (same pattern as test_calib_debug) ──
static const std::string CALIB_PROJECT = "/Users/johnsonr/red_data/test_calib1/";
static const std::string POINTSOURCE_PROJECT = "/Users/johnsonr/red_data/test_pointsource2/";

// ── CameraPose (matches CalibrationPipeline::CameraPose) ──
struct CameraPose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 5, 1> dist = Eigen::Matrix<double, 5, 1>::Zero();
};

// ── Load a camera YAML file into CameraPose ──
static CameraPose load_pose(const std::string &yaml_path) {
    auto yaml = opencv_yaml::read(yaml_path);
    CameraPose pose;
    pose.K = yaml.getMatrix("camera_matrix").block<3, 3>(0, 0);
    Eigen::MatrixXd dist_mat = yaml.getMatrix("distortion_coefficients");
    for (int j = 0; j < 5; j++)
        pose.dist(j) = dist_mat(j, 0);
    pose.R = yaml.getMatrix("rc_ext").block<3, 3>(0, 0);
    Eigen::MatrixXd t_mat = yaml.getMatrix("tc_ext");
    pose.t = Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));
    return pose;
}

// ── Find latest timestamped directory ──
static std::string find_latest_dir(const std::string &parent) {
    std::string latest;
    for (const auto &entry : fs::directory_iterator(parent)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            if (name > latest)
                latest = name;
        }
    }
    return parent + "/" + latest;
}

// ── Per-camera stats ──
struct CameraStats {
    std::string serial;
    int n_pts = 0;
    double sum_err_orig = 0.0;
    double sum_err_laser = 0.0;
};

// ── Triangulate + reproject for one set of poses ──
// Returns per-camera reprojection errors for each (landmark_id, camera) pair.
// Also populates per-camera point counts and error sums in stats_map.
static void evaluate_poses(
    const std::map<std::string, CameraPose> &poses,
    const std::map<int, std::vector<std::pair<std::string, Eigen::Vector2d>>> &observations,
    std::map<std::string, double> &cam_sum_err,
    std::map<std::string, int> &cam_n_pts,
    int &total_pts, double &total_err,
    int min_views = 2)
{
    for (const auto &[lid, obs] : observations) {
        if ((int)obs.size() < min_views)
            continue;

        // Build undistorted points and projection matrices for triangulation
        std::vector<Eigen::Vector2d> undist_pts;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mats;
        std::vector<std::string> cam_serials;
        std::vector<Eigen::Vector2d> orig_pixels;

        for (const auto &[serial, pixel] : obs) {
            auto it = poses.find(serial);
            if (it == poses.end())
                continue;
            const CameraPose &p = it->second;

            // Undistort the 2D detection
            Eigen::Vector2d undist = red_math::undistortPoint(pixel, p.K, p.dist);
            undist_pts.push_back(undist);

            // Projection matrix P = K * [R | t]
            proj_mats.push_back(red_math::projectionFromKRt(p.K, p.R, p.t));

            cam_serials.push_back(serial);
            orig_pixels.push_back(pixel);
        }

        if ((int)undist_pts.size() < min_views)
            continue;

        // Triangulate
        Eigen::Vector3d pt3d = red_math::triangulatePoints(undist_pts, proj_mats);

        // Reproject into each camera and measure error vs original detection
        for (size_t i = 0; i < cam_serials.size(); i++) {
            const CameraPose &p = poses.at(cam_serials[i]);
            Eigen::Vector3d rvec = red_math::rotationMatrixToVector(p.R);
            Eigen::Vector2d proj = red_math::projectPoint(pt3d, rvec, p.t, p.K, p.dist);
            double err = (proj - orig_pixels[i]).norm();

            cam_sum_err[cam_serials[i]] += err;
            cam_n_pts[cam_serials[i]]++;
            total_err += err;
            total_pts++;
        }
    }
}

int main() {
    // ── Locate laser calibration output (latest timestamped dir) ──
    std::string pointsource_calib_parent = POINTSOURCE_PROJECT + "pointsource_calibration/";
    if (!fs::exists(pointsource_calib_parent)) {
        std::cerr << "Laser calibration dir not found: " << pointsource_calib_parent << "\n";
        return 1;
    }
    std::string pointsource_calib_dir = find_latest_dir(pointsource_calib_parent);

    std::string calib_dir = CALIB_PROJECT + "calibration/";
    std::string landmarks_path = CALIB_PROJECT + "output/landmarks.json";

    std::cout << "=== Calibration Cross-Validation (triangulate-and-reproject) ===\n";
    std::cout << "Calib project: " << CALIB_PROJECT << "\n";
    std::cout << "Laser project: " << pointsource_calib_dir << "/\n\n";

    // ── Load landmarks.json ──
    // Format: { "serial": { "ids": [int...], "landmarks": [[x,y]...] } }
    json landmarks_json;
    {
        std::ifstream f(landmarks_path);
        if (!f.is_open()) {
            std::cerr << "Cannot open " << landmarks_path << "\n";
            return 1;
        }
        f >> landmarks_json;
    }

    // ── Build observation map: landmark_id → [(serial, pixel_2d), ...] ──
    std::map<int, std::vector<std::pair<std::string, Eigen::Vector2d>>> observations;
    int total_obs = 0;

    std::vector<std::string> serials;
    for (auto &[serial, _] : landmarks_json.items())
        serials.push_back(serial);
    std::sort(serials.begin(), serials.end());

    for (const auto &serial : serials) {
        const auto &cam_data = landmarks_json[serial];
        const auto &ids = cam_data["ids"];
        const auto &lms = cam_data["landmarks"];

        for (size_t i = 0; i < ids.size(); i++) {
            int lid = ids[i].get<int>();
            Eigen::Vector2d pixel(lms[i][0].get<double>(), lms[i][1].get<double>());
            observations[lid].emplace_back(serial, pixel);
            total_obs++;
        }
    }

    // Count landmarks with 2+ views
    int n_triangulable = 0;
    for (const auto &[lid, obs] : observations)
        if (obs.size() >= 2) n_triangulable++;

    std::cout << "Total observations: " << total_obs << " across " << serials.size() << " cameras\n";
    std::cout << "Unique landmarks: " << observations.size()
              << " (" << n_triangulable << " with 2+ views)\n";

    // ── Load poses for both calibrations ──
    std::map<std::string, CameraPose> orig_poses, pointsource_poses;
    for (const auto &serial : serials) {
        std::string orig_yaml = calib_dir + "Cam" + serial + ".yaml";
        std::string laser_yaml = pointsource_calib_dir + "/Cam" + serial + ".yaml";

        if (fs::exists(orig_yaml))
            orig_poses[serial] = load_pose(orig_yaml);
        else
            std::cerr << "Warning: missing " << orig_yaml << "\n";

        if (fs::exists(laser_yaml))
            pointsource_poses[serial] = load_pose(laser_yaml);
        else
            std::cerr << "Warning: missing " << laser_yaml << "\n";
    }

    // ── Evaluate both sets of poses ──
    std::map<std::string, double> orig_cam_err, pointsource_cam_err;
    std::map<std::string, int> orig_cam_n, laser_cam_n;
    int orig_total_pts = 0, laser_total_pts = 0;
    double orig_total_err = 0.0, laser_total_err = 0.0;

    evaluate_poses(orig_poses, observations,
                   orig_cam_err, orig_cam_n, orig_total_pts, orig_total_err);
    evaluate_poses(pointsource_poses, observations,
                   pointsource_cam_err, laser_cam_n, laser_total_pts, laser_total_err);

    // ── Print results ──
    std::cout << "\n";
    std::cout << std::left << std::setw(14) << "Camera"
              << std::right << std::setw(8) << "N_pts"
              << std::setw(12) << "Orig(px)"
              << std::setw(12) << "Laser(px)"
              << std::setw(12) << "Delta"
              << "\n";
    std::cout << std::string(58, '-') << "\n";

    for (const auto &serial : serials) {
        int n_orig = orig_cam_n.count(serial) ? orig_cam_n[serial] : 0;
        int n_laser = laser_cam_n.count(serial) ? laser_cam_n[serial] : 0;
        if (n_orig == 0 && n_laser == 0)
            continue;

        double mean_orig = n_orig > 0 ? orig_cam_err[serial] / n_orig : 0.0;
        double mean_laser = n_laser > 0 ? pointsource_cam_err[serial] / n_laser : 0.0;
        double delta = mean_laser - mean_orig;

        std::cout << std::left << std::setw(14) << serial
                  << std::right << std::setw(8) << n_orig
                  << std::setw(12) << std::fixed << std::setprecision(3) << mean_orig
                  << std::setw(12) << mean_laser
                  << std::setw(12) << std::showpos << delta << std::noshowpos
                  << "\n";
    }

    if (orig_total_pts > 0) {
        double mean_orig = orig_total_err / orig_total_pts;
        double mean_laser = laser_total_err / laser_total_pts;
        double delta = mean_laser - mean_orig;

        std::cout << std::string(58, '-') << "\n";
        std::cout << std::left << std::setw(14) << "TOTAL"
                  << std::right << std::setw(8) << orig_total_pts
                  << std::setw(12) << std::fixed << std::setprecision(3) << mean_orig
                  << std::setw(12) << mean_laser
                  << std::setw(12) << std::showpos << delta << std::noshowpos
                  << "\n";
    }

    return 0;
}

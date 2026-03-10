#pragma once
/*  jarvis_import.h  — Import JARVIS predictions into RED label format
 *
 *  Reads JARVIS data3D.csv (3D keypoints + per-keypoint confidence),
 *  projects to 2D per camera, and writes RED-format label CSVs that
 *  load_keypoints() can read directly.
 */

#include "camera.h"
#include "red_math.h"
#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace JarvisImport {

struct Prediction3D {
    std::vector<Eigen::Vector3d> positions;   // [num_joints]
    std::vector<float> confidences;           // [num_joints], 0-1
};

// ---------------------------------------------------------------------------
// Read JARVIS data3D.csv
// Format: 2 header rows, then data rows with 4 cols per keypoint (x,y,z,conf)
// Frame ID is implicit (row index starting from 0 after headers)
// Rows containing "NaN" are skipped
// ---------------------------------------------------------------------------
inline std::map<int, Prediction3D>
read_jarvis_predictions(const std::string &csv_path,
                        float conf_threshold = 0.0f,
                        std::string *error = nullptr) {
    std::map<int, Prediction3D> result;

    std::ifstream fin(csv_path);
    if (!fin) {
        if (error) *error = "Failed to open: " + csv_path;
        return result;
    }

    std::string line;
    int line_num = 0;
    int frame_id = 0;

    while (std::getline(fin, line)) {
        // Skip 2 header rows
        if (line_num < 2) { line_num++; continue; }

        // Skip rows containing NaN
        if (line.find("NaN") != std::string::npos ||
            line.find("nan") != std::string::npos) {
            frame_id++;
            line_num++;
            continue;
        }

        // Parse comma-separated values
        std::vector<double> values;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            token = token.substr(start);
            try {
                values.push_back(std::stod(token));
            } catch (...) {
                break; // skip malformed values
            }
        }

        // Must have groups of 4 (x, y, z, confidence)
        int num_joints = (int)values.size() / 4;
        if (num_joints > 0 && values.size() == (size_t)num_joints * 4) {
            Prediction3D pred;
            float conf_sum = 0;
            for (int j = 0; j < num_joints; j++) {
                pred.positions.push_back(Eigen::Vector3d(
                    values[j*4+0], values[j*4+1], values[j*4+2]));
                float c = (float)values[j*4+3];
                pred.confidences.push_back(c);
                conf_sum += c;
            }

            float mean_conf = conf_sum / num_joints;
            if (mean_conf >= conf_threshold) {
                result[frame_id] = std::move(pred);
            }
        }

        frame_id++;
        line_num++;
    }

    return result;
}

// ---------------------------------------------------------------------------
// Project 3D predictions to 2D for a single camera
// Returns 2D points in image coordinates (top-left origin)
// ---------------------------------------------------------------------------
inline std::map<int, std::vector<Eigen::Vector2d>>
project_to_camera(const std::map<int, Prediction3D> &preds,
                  const CameraParams &cam) {
    std::map<int, std::vector<Eigen::Vector2d>> result;
    for (const auto &[fid, pred] : preds) {
        std::vector<Eigen::Vector2d> pts2d;
        pts2d.reserve(pred.positions.size());
        for (const auto &pt3d : pred.positions) {
            Eigen::Vector2d pt2d = red_math::projectPoint(
                pt3d, cam.rvec, cam.tvec, cam.k, cam.dist_coeffs);
            pts2d.push_back(pt2d);
        }
        result[fid] = std::move(pts2d);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Write RED-format label CSVs:
//   keypoints3d.csv + <camera>.csv per camera + confidence.csv
// Output matches save_keypoints() format so load_keypoints() can read it.
// ---------------------------------------------------------------------------
inline bool
write_prediction_csvs(const std::string &output_folder,
                      const std::string &skeleton_name,
                      const std::map<int, Prediction3D> &preds,
                      const std::vector<CameraParams> &cameras,
                      const std::vector<std::string> &camera_names,
                      int img_height,
                      std::string *error = nullptr) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(output_folder, ec);
    if (ec) {
        if (error) *error = "Failed to create directory: " + output_folder;
        return false;
    }

    // --- keypoints3d.csv ---
    {
        std::ofstream f(output_folder + "/keypoints3d.csv");
        if (!f) {
            if (error) *error = "Failed to write keypoints3d.csv";
            return false;
        }
        f << skeleton_name << "\n";
        for (const auto &[fid, pred] : preds) {
            f << fid;
            for (int j = 0; j < (int)pred.positions.size(); j++) {
                f << "," << j
                  << "," << pred.positions[j].x()
                  << "," << pred.positions[j].y()
                  << "," << pred.positions[j].z();
            }
            f << "\n";
        }
    }

    // --- Per-camera 2D CSVs ---
    for (int c = 0; c < (int)cameras.size(); c++) {
        auto pts2d_map = project_to_camera(preds, cameras[c]);
        std::ofstream f(output_folder + "/" + camera_names[c] + ".csv");
        if (!f) {
            if (error) *error = "Failed to write " + camera_names[c] + ".csv";
            return false;
        }
        f << skeleton_name << "\n";
        for (const auto &[fid, pts2d] : pts2d_map) {
            f << fid;
            for (int j = 0; j < (int)pts2d.size(); j++) {
                // Y-flip: image coords (top-left) → ImPlot coords (bottom-left)
                double x = pts2d[j].x();
                double y = (double)img_height - pts2d[j].y();
                f << "," << j << "," << x << "," << y;
            }
            f << "\n";
        }
    }

    // --- confidence.csv ---
    {
        std::ofstream f(output_folder + "/confidence.csv");
        if (!f) {
            if (error) *error = "Failed to write confidence.csv";
            return false;
        }
        f << "frame_id";
        if (!preds.empty()) {
            int nj = (int)preds.begin()->second.confidences.size();
            for (int j = 0; j < nj; j++) f << ",kp" << j;
        }
        f << "\n";
        for (const auto &[fid, pred] : preds) {
            f << fid;
            for (float c : pred.confidences) f << "," << c;
            f << "\n";
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// Load confidence.csv into keypoints_map (companion to load_keypoints)
// ---------------------------------------------------------------------------
inline int
load_confidence(const std::string &folder,
                std::map<uint32_t, struct KeyPoints *> &keypoints_map,
                int num_cameras, int num_nodes) {
    std::string path = folder + "/confidence.csv";
    std::ifstream fin(path);
    if (!fin) return 0; // not an error — file is optional

    std::string line;
    int line_num = 0;
    int loaded = 0;
    while (std::getline(fin, line)) {
        if (line_num++ == 0) continue; // skip header

        std::stringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        uint32_t frame_id = (uint32_t)std::stoul(token);

        auto it = keypoints_map.find(frame_id);
        if (it == keypoints_map.end()) continue;

        auto *kp = it->second;
        for (int j = 0; j < num_nodes; j++) {
            if (!std::getline(ss, token, ',')) break;
            float conf = std::stof(token);
            kp->kp3d[j].confidence = conf;
            for (int c = 0; c < num_cameras; c++) {
                kp->kp2d[c][j].confidence = conf;
            }
        }
        loaded++;
    }
    return loaded;
}

// ---------------------------------------------------------------------------
// Full import: read data3D.csv → write RED CSVs → ready for load_keypoints()
// ---------------------------------------------------------------------------
struct ImportResult {
    int frames_imported = 0;
    int frames_filtered = 0;
    int num_keypoints = 0;
    float mean_confidence = 0;
    std::string output_folder;
    std::string error;
};

inline ImportResult
import_jarvis_predictions(const std::string &data3d_csv_path,
                          const std::string &predictions_root,
                          const std::string &skeleton_name,
                          const std::vector<CameraParams> &cameras,
                          const std::vector<std::string> &camera_names,
                          int img_height,
                          float conf_threshold = 0.0f) {
    ImportResult result;

    // Read predictions
    auto all_preds = read_jarvis_predictions(data3d_csv_path, 0.0f,
                                              &result.error);
    if (!result.error.empty()) return result;

    int total = (int)all_preds.size();

    // Filter by confidence
    std::map<int, Prediction3D> filtered;
    float conf_sum = 0;
    for (auto &[fid, pred] : all_preds) {
        float mean = 0;
        for (float c : pred.confidences) mean += c;
        mean /= (float)pred.confidences.size();
        if (mean >= conf_threshold) {
            conf_sum += mean;
            filtered[fid] = std::move(pred);
        }
    }

    result.frames_imported = (int)filtered.size();
    result.frames_filtered = total - result.frames_imported;
    if (!filtered.empty()) {
        result.num_keypoints = (int)filtered.begin()->second.positions.size();
        result.mean_confidence = conf_sum / (float)filtered.size();
    }

    // Generate timestamped output folder
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S", std::localtime(&t));
    result.output_folder = predictions_root + "/" + std::string(buf);

    // Write CSVs
    if (!write_prediction_csvs(result.output_folder, skeleton_name,
                                filtered, cameras, camera_names,
                                img_height, &result.error)) {
        return result;
    }

    return result;
}

} // namespace JarvisImport

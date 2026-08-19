#pragma once
/*  jarvis_import.h  — Import JARVIS predictions into RED label format
 *
 *  Reads JARVIS data3D.csv (3D keypoints + per-keypoint confidence) into
 *  Prediction3D records. Callers decide where they go: gui/jarvis_import_window.h
 *  streams them into a .rpred prediction store, or reprojects them straight
 *  into the AnnotationMap as editable labels.
 */

#include <Eigen/Core>
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

} // namespace JarvisImport

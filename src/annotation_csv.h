#pragma once
// annotation_csv.h -- CSV I/O for the annotation system (v2 format)
//
// All CSV persistence for AnnotationMap lives here: save, load,
// v1-to-v2 conversion, and find_most_recent_labels.
//
// CSV v2 format:
//   #red_csv v2
//   #skeleton <name>
//   frame,x0,y0,c0,s0,x1,y1,c1,s1,...       (2D per-camera)
//   frame,x0,y0,z0,c0,x1,y1,z1,c1,...       (3D)
//
// Empty cells = unlabeled (no 1E7 sentinel in file).
// c = confidence (empty = manual, float = predicted).
// s = source flag (empty = Manual, P = Predicted, I = Imported).
// Coordinates in ImPlot space (Y=0 at bottom).

#include "annotation.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace AnnotationCSV {

// =========================================================================
// Parser primitives
// =========================================================================

// Parse the next CSV cell as a double.
// On entry, ptr points to the beginning of the cell.
// If the cell is empty (next char is comma, newline, or null), advances past
// the comma (if present) and returns false. Otherwise, reads via strtod,
// advances ptr past the trailing comma, and returns true.
inline bool parse_csv_double(const char *&ptr, double &out) {
    // Skip leading whitespace
    while (*ptr == ' ' || *ptr == '\t') ++ptr;

    // Empty cell?
    if (*ptr == ',' || *ptr == '\n' || *ptr == '\r' || *ptr == '\0') {
        if (*ptr == ',') ++ptr;
        return false;
    }

    char *end = nullptr;
    out = std::strtod(ptr, &end);
    if (end == ptr) {
        // strtod consumed nothing (unparseable) — skip to next comma or end
        while (*ptr != ',' && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') ++ptr;
        if (*ptr == ',') ++ptr;
        return false;
    }
    ptr = end;
    if (*ptr == ',') ++ptr;
    return true;
}

// Parse the next CSV cell as a single character (source flag).
// Returns the character, or '\0' if the cell is empty.
// Advances past the trailing comma.
inline char parse_csv_char(const char *&ptr) {
    // Skip leading whitespace
    while (*ptr == ' ' || *ptr == '\t') ++ptr;

    // Empty cell?
    if (*ptr == ',' || *ptr == '\n' || *ptr == '\r' || *ptr == '\0') {
        if (*ptr == ',') ++ptr;
        return '\0';
    }

    char ch = *ptr;
    ++ptr;
    // Advance to comma or end
    while (*ptr != ',' && *ptr != '\n' && *ptr != '\r' && *ptr != '\0') ++ptr;
    if (*ptr == ',') ++ptr;
    return ch;
}

// =========================================================================
// Timestamp helper
// =========================================================================

// Returns current time as "YYYY_MM_DD_HH_MM_SS".
inline std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S", std::localtime(&t));
    return std::string(buf);
}

// =========================================================================
// Save functions
// =========================================================================

// Write v2 2D CSV for a single camera.
// Path should be e.g. "<folder>/camera1.csv".
inline bool save_2d_csv(const std::string &path, const std::string &skeleton_name,
                         const AnnotationMap &amap, int cam_idx, int num_nodes) {
    std::ofstream f(path);
    if (!f) return false;

    // Header lines
    f << "#red_csv v2\n";
    f << "#skeleton " << skeleton_name << "\n";

    // Column header: frame, then groups of (x, y, c, s) per keypoint
    f << "frame";
    for (int k = 0; k < num_nodes; ++k)
        f << ",x" << k << ",y" << k << ",c" << k << ",s" << k;
    f << "\n";

    // Data rows
    for (const auto &[frame, fa] : amap) {
        if (cam_idx >= (int)fa.cameras.size()) continue;
        const auto &cam = fa.cameras[cam_idx];

        f << frame;
        for (int k = 0; k < num_nodes; ++k) {
            if (k < (int)cam.keypoints.size() && cam.keypoints[k].labeled) {
                const auto &kp = cam.keypoints[k];
                f << "," << kp.x << "," << kp.y << ",";
                if (kp.confidence > 0.0f)
                    f << kp.confidence;
                f << ",";
                if (kp.source == LabelSource::Predicted) f << "P";
                else if (kp.source == LabelSource::Imported) f << "I";
            } else {
                f << ",,,,";
            }
        }
        f << "\n";
    }
    return true;
}

// Write v2 3D CSV.
// Path should be e.g. "<folder>/keypoints3d.csv".
inline bool save_3d_csv(const std::string &path, const std::string &skeleton_name,
                         const AnnotationMap &amap, int num_nodes) {
    std::ofstream f(path);
    if (!f) return false;

    // Header lines
    f << "#red_csv v2\n";
    f << "#skeleton " << skeleton_name << "\n";

    // Column header: frame, then groups of (x, y, z, c) per keypoint
    f << "frame";
    for (int k = 0; k < num_nodes; ++k)
        f << ",x" << k << ",y" << k << ",z" << k << ",c" << k;
    f << "\n";

    // Data rows
    for (const auto &[frame, fa] : amap) {
        f << frame;
        for (int k = 0; k < num_nodes; ++k) {
            if (k < (int)fa.kp3d.size() && fa.kp3d[k].triangulated) {
                const auto &kp = fa.kp3d[k];
                f << "," << kp.x << "," << kp.y << "," << kp.z << ",";
                if (kp.confidence > 0.0f)
                    f << kp.confidence;
            } else {
                f << ",,,,";
            }
        }
        f << "\n";
    }
    return true;
}

// Create a timestamped subfolder under root_dir, save 3D + all 2D CSVs.
// Returns the folder path on success, or empty string on error.
inline std::string save_all(const std::string &root_dir, const std::string &skeleton_name,
                             const AnnotationMap &amap, int num_cameras, int num_nodes,
                             const std::vector<std::string> &camera_names,
                             std::string *error = nullptr) {
    std::string ts = current_timestamp();
    std::string folder = root_dir + "/" + ts;

    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    if (ec) {
        if (error) *error = "Failed to create directory: " + folder;
        return {};
    }

    // Save 3D
    if (!save_3d_csv(folder + "/keypoints3d.csv", skeleton_name, amap, num_nodes)) {
        if (error) *error = "Failed to write keypoints3d.csv";
        return {};
    }

    // Save 2D per camera
    for (int c = 0; c < num_cameras; ++c) {
        std::string cam_name = (c < (int)camera_names.size())
                                    ? camera_names[c]
                                    : "camera" + std::to_string(c);
        std::string cam_path = folder + "/" + cam_name + ".csv";
        if (!save_2d_csv(cam_path, skeleton_name, amap, c, num_nodes)) {
            if (error) *error = "Failed to write " + cam_name + ".csv";
            return {};
        }
    }

    // Save extended annotations (bbox, obb, mask) if any
    save_annotations_json(amap, folder);

    return folder;
}

// =========================================================================
// Load functions
// =========================================================================

// Load v2 3D CSV into an AnnotationMap.
// Creates FrameAnnotation entries via get_or_create_frame.
inline bool load_3d_csv(const std::string &path, AnnotationMap &amap,
                         int num_nodes, int num_cameras, std::string &error) {
    std::ifstream fin(path);
    if (!fin) {
        error = "Failed to open: " + path;
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        // Skip comment lines
        if (!line.empty() && line[0] == '#') continue;
        // Skip column header
        if (line.size() >= 6 && line.substr(0, 6) == "frame,") continue;

        if (line.empty()) continue;

        const char *ptr = line.c_str();
        double frame_d;
        if (!parse_csv_double(ptr, frame_d)) continue;
        u32 frame = (u32)frame_d;

        FrameAnnotation &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);

        // Read groups of (x, y, z, c) per keypoint
        for (int k = 0; k < num_nodes; ++k) {
            double x, y, z, c;
            bool has_x = parse_csv_double(ptr, x);
            bool has_y = parse_csv_double(ptr, y);
            bool has_z = parse_csv_double(ptr, z);
            bool has_c = parse_csv_double(ptr, c);

            if (has_x && has_y && has_z) {
                fa.kp3d[k].x = x;
                fa.kp3d[k].y = y;
                fa.kp3d[k].z = z;
                fa.kp3d[k].triangulated = true;
                fa.kp3d[k].confidence = has_c ? (float)c : 0.0f;
            }
            // else: stays at default (UNLABELED, triangulated=false)
        }
    }
    return true;
}

// Load v2 2D CSV for a single camera into an AnnotationMap.
// Writes to amap[frame].cameras[cam_idx].
inline bool load_2d_csv(const std::string &path, AnnotationMap &amap,
                         int cam_idx, int num_nodes, int num_cameras,
                         std::string &error) {
    std::ifstream fin(path);
    if (!fin) {
        error = "Failed to open: " + path;
        return false;
    }

    std::string line;
    while (std::getline(fin, line)) {
        // Skip comment lines
        if (!line.empty() && line[0] == '#') continue;
        // Skip column header
        if (line.size() >= 6 && line.substr(0, 6) == "frame,") continue;

        if (line.empty()) continue;

        const char *ptr = line.c_str();
        double frame_d;
        if (!parse_csv_double(ptr, frame_d)) continue;
        u32 frame = (u32)frame_d;

        FrameAnnotation &fa = get_or_create_frame(amap, frame, num_nodes, num_cameras);
        if (cam_idx >= (int)fa.cameras.size()) continue;
        auto &cam = fa.cameras[cam_idx];

        // Read groups of (x, y, c, s) per keypoint
        for (int k = 0; k < num_nodes; ++k) {
            double x, y, c;
            bool has_x = parse_csv_double(ptr, x);
            bool has_y = parse_csv_double(ptr, y);
            bool has_c = parse_csv_double(ptr, c);
            char  src  = parse_csv_char(ptr);

            if (has_x && has_y) {
                cam.keypoints[k].x = x;
                cam.keypoints[k].y = y;
                cam.keypoints[k].labeled = true;
                cam.keypoints[k].confidence = has_c ? (float)c : 0.0f;
                if (src == 'P') cam.keypoints[k].source = LabelSource::Predicted;
                else if (src == 'I') cam.keypoints[k].source = LabelSource::Imported;
                else cam.keypoints[k].source = LabelSource::Manual;
            }
            // else: stays at default (UNLABELED, labeled=false)
        }
    }
    return true;
}

// Load a complete v2 label folder (3D + all 2D per-camera).
// Loads 3D first (single-threaded), then loads 2D per-camera in parallel.
// Returns 0 on success, 1 on error.
inline int load_all(const std::string &folder, AnnotationMap &amap,
                     const std::string &skeleton_name, int num_nodes, int num_cameras,
                     const std::vector<std::string> &camera_names, std::string &error) {
    namespace fs = std::filesystem;

    // Validate skeleton from 3D header
    std::string kp3d_path = folder + "/keypoints3d.csv";
    if (fs::exists(kp3d_path)) {
        // Optionally check skeleton name in header
        std::ifstream hdr(kp3d_path);
        std::string first_line;
        if (std::getline(hdr, first_line)) {
            if (first_line.find("#red_csv") != std::string::npos) {
                // v2 format, check skeleton line
                std::string skel_line;
                if (std::getline(hdr, skel_line)) {
                    if (skel_line.find("#skeleton") != std::string::npos && skel_line.size() > 10) {
                        std::string skel_in_file = skel_line.substr(10); // after "#skeleton "
                        // Trim whitespace
                        while (!skel_in_file.empty() && (skel_in_file.back() == ' ' ||
                               skel_in_file.back() == '\r' || skel_in_file.back() == '\n'))
                            skel_in_file.pop_back();
                        if (!skeleton_name.empty() && skel_in_file != skeleton_name) {
                            error = "Skeleton mismatch: file has '" + skel_in_file +
                                    "', project has '" + skeleton_name + "'";
                            return 1;
                        }
                    }
                }
            }
        }

        // Load 3D (single-threaded)
        if (!load_3d_csv(kp3d_path, amap, num_nodes, num_cameras, error))
            return 1;
    }

    // Pre-scan 2D CSVs for frame numbers and pre-create all entries
    // on the main thread. This ensures no concurrent map insertion.
    for (int c = 0; c < num_cameras; ++c) {
        std::string cam_name = (c < (int)camera_names.size())
                                    ? camera_names[c]
                                    : "camera" + std::to_string(c);
        std::string cam_path = folder + "/" + cam_name + ".csv";
        if (!fs::exists(cam_path)) continue;
        std::ifstream scan(cam_path);
        std::string line;
        while (std::getline(scan, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.size() >= 6 && line.substr(0, 6) == "frame,") continue;
            const char *p = line.c_str();
            double fd;
            if (parse_csv_double(p, fd))
                get_or_create_frame(amap, (u32)fd, num_nodes, num_cameras);
        }
    }

    // Load 2D per-camera in parallel (all frames pre-created, no map mutations)
    std::vector<std::thread> threads;
    std::vector<std::string> errors(num_cameras);
    std::vector<bool> results(num_cameras, true);

    for (int c = 0; c < num_cameras; ++c) {
        std::string cam_name = (c < (int)camera_names.size())
                                    ? camera_names[c]
                                    : "camera" + std::to_string(c);
        std::string cam_path = folder + "/" + cam_name + ".csv";

        if (!fs::exists(cam_path)) continue;

        threads.emplace_back(
            [&amap, cam_path, c, num_nodes, num_cameras, &errors, &results]() {
                results[c] = load_2d_csv(cam_path, amap, c, num_nodes,
                                          num_cameras, errors[c]);
            });
    }

    for (auto &t : threads) t.join();

    // Check for errors
    bool has_error = false;
    for (int c = 0; c < num_cameras; ++c) {
        if (!results[c]) {
            std::string cam_name = (c < (int)camera_names.size())
                                        ? camera_names[c]
                                        : "camera" + std::to_string(c);
            error += cam_name + ": " + errors[c] + "\n";
            has_error = true;
        }
    }

    // Load extended annotations (bbox, obb, mask) if present
    load_annotations_json(amap, folder);

    return has_error ? 1 : 0;
}

// =========================================================================
// V1 detection and conversion
// =========================================================================

// Check whether a label folder uses v1 format (no "#red_csv" header).
// Reads the first line of keypoints3d.csv or the first camera CSV.
inline bool is_v1_format(const std::string &folder,
                          const std::vector<std::string> &camera_names) {
    namespace fs = std::filesystem;

    // Try keypoints3d.csv first
    std::string kp3d_path = folder + "/keypoints3d.csv";
    if (fs::exists(kp3d_path)) {
        std::ifstream f(kp3d_path);
        std::string line;
        if (std::getline(f, line)) {
            return line.find("#red_csv") == std::string::npos;
        }
    }

    // Try first camera CSV
    for (const auto &name : camera_names) {
        std::string cam_path = folder + "/" + name + ".csv";
        if (fs::exists(cam_path)) {
            std::ifstream f(cam_path);
            std::string line;
            if (std::getline(f, line)) {
                return line.find("#red_csv") == std::string::npos;
            }
        }
    }

    return false; // no files found, not v1
}

// Convert a v1 label folder to v2 format.
//
// V1 3D format: line 0 = skeleton name, then "frame,idx,x,y,z,idx,x,y,z,..."
//   (groups of 4: idx, x, y, z)
// V1 2D format: line 0 = skeleton name, then "frame,idx,x,y,idx,x,y,..."
//   (groups of 3: idx, x, y)
// 1E7 sentinel means unlabeled.
// confidence.csv (optional): "frame_id,kp0,kp1,..." with per-keypoint floats.
inline bool convert_v1_to_v2(const std::string &src_folder, const std::string &dst_folder,
                               const std::string &skeleton_name, int num_nodes,
                               const std::vector<std::string> &camera_names,
                               std::string *error) {
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::create_directories(dst_folder, ec);
    if (ec) {
        if (error) *error = "Failed to create directory: " + dst_folder;
        return false;
    }

    // Helper: check if a value is the v1 "unlabeled" sentinel
    auto is_sentinel = [](double v) { return std::abs(v) > 9.99e6; };

    // ---- Load optional confidence.csv ----
    // Map: frame_id -> vector<float> per keypoint
    std::map<u32, std::vector<float>> confidence_map;
    {
        std::string conf_path = src_folder + "/confidence.csv";
        std::ifstream cfin(conf_path);
        if (cfin) {
            std::string line;
            int line_num = 0;
            while (std::getline(cfin, line)) {
                if (line_num++ == 0) continue; // skip header
                if (line.empty()) continue;
                std::stringstream ss(line);
                std::string token;
                std::getline(ss, token, ',');
                u32 fid = (u32)std::stoul(token);
                std::vector<float> confs;
                while (std::getline(ss, token, ',')) {
                    try { confs.push_back(std::stof(token)); }
                    catch (...) { confs.push_back(0.0f); }
                }
                confidence_map[fid] = std::move(confs);
            }
        }
    }

    // ---- Convert keypoints3d.csv ----
    {
        std::string src_path = src_folder + "/keypoints3d.csv";
        std::string dst_path = dst_folder + "/keypoints3d.csv";
        std::ifstream fin(src_path);
        if (fin) {
            std::ofstream fout(dst_path);
            if (!fout) {
                if (error) *error = "Failed to write: " + dst_path;
                return false;
            }

            // Write v2 header
            fout << "#red_csv v2\n";
            fout << "#skeleton " << skeleton_name << "\n";
            fout << "frame";
            for (int k = 0; k < num_nodes; ++k)
                fout << ",x" << k << ",y" << k << ",z" << k << ",c" << k;
            fout << "\n";

            std::string line;
            int line_num = 0;
            while (std::getline(fin, line)) {
                if (line_num++ == 0) continue; // skip skeleton name line
                if (line.empty()) continue;

                // Strip trailing whitespace / carriage return
                while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
                    line.pop_back();
                if (line.empty()) continue;

                // Parse: frame,idx,x,y,z,idx,x,y,z,...
                std::stringstream ss(line);
                std::string token;
                std::getline(ss, token, ',');
                u32 frame = (u32)std::stoul(token);

                // Read all remaining values
                std::vector<double> vals;
                while (std::getline(ss, token, ',')) {
                    try { vals.push_back(std::stod(token)); }
                    catch (...) { vals.push_back(UNLABELED); }
                }

                // Parse groups of 4 (idx, x, y, z)
                // Build per-keypoint arrays
                std::vector<double> px(num_nodes, UNLABELED);
                std::vector<double> py(num_nodes, UNLABELED);
                std::vector<double> pz(num_nodes, UNLABELED);
                std::vector<bool>   labeled(num_nodes, false);

                for (size_t i = 0; i + 3 < vals.size(); i += 4) {
                    int idx = (int)vals[i];
                    if (idx < 0 || idx >= num_nodes) continue;
                    double x = vals[i+1], y = vals[i+2], z = vals[i+3];
                    if (!is_sentinel(x) && !is_sentinel(y) && !is_sentinel(z)) {
                        px[idx] = x; py[idx] = y; pz[idx] = z;
                        labeled[idx] = true;
                    }
                }

                // Look up confidence
                auto cit = confidence_map.find(frame);

                fout << frame;
                for (int k = 0; k < num_nodes; ++k) {
                    fout << ",";
                    if (labeled[k]) {
                        fout << px[k] << "," << py[k] << "," << pz[k] << ",";
                        // Write confidence if available and > 0
                        if (cit != confidence_map.end() &&
                            k < (int)cit->second.size() && cit->second[k] > 0.0f) {
                            fout << cit->second[k];
                        }
                    } else {
                        fout << ",,,";
                    }
                }
                fout << "\n";
            }
        }
    }

    // ---- Convert per-camera 2D CSVs ----
    for (const auto &cam_name : camera_names) {
        std::string src_path = src_folder + "/" + cam_name + ".csv";
        std::string dst_path = dst_folder + "/" + cam_name + ".csv";
        std::ifstream fin(src_path);
        if (!fin) continue; // camera file may not exist

        std::ofstream fout(dst_path);
        if (!fout) {
            if (error) *error = "Failed to write: " + dst_path;
            return false;
        }

        // Write v2 header
        fout << "#red_csv v2\n";
        fout << "#skeleton " << skeleton_name << "\n";
        fout << "frame";
        for (int k = 0; k < num_nodes; ++k)
            fout << ",x" << k << ",y" << k << ",c" << k << ",s" << k;
        fout << "\n";

        std::string line;
        int line_num = 0;
        while (std::getline(fin, line)) {
            if (line_num++ == 0) continue; // skip skeleton name line
            if (line.empty()) continue;

            while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
                line.pop_back();
            if (line.empty()) continue;

            // Parse: frame,idx,x,y,idx,x,y,...
            std::stringstream ss(line);
            std::string token;
            std::getline(ss, token, ',');
            u32 frame = (u32)std::stoul(token);

            // Read all remaining values
            std::vector<double> vals;
            while (std::getline(ss, token, ',')) {
                try { vals.push_back(std::stod(token)); }
                catch (...) { vals.push_back(UNLABELED); }
            }

            // Parse groups of 3 (idx, x, y)
            std::vector<double> px(num_nodes, UNLABELED);
            std::vector<double> py(num_nodes, UNLABELED);
            std::vector<bool>   labeled(num_nodes, false);

            for (size_t i = 0; i + 2 < vals.size(); i += 3) {
                int idx = (int)vals[i];
                if (idx < 0 || idx >= num_nodes) continue;
                double x = vals[i+1], y = vals[i+2];
                if (!is_sentinel(x) && !is_sentinel(y)) {
                    px[idx] = x; py[idx] = y;
                    labeled[idx] = true;
                }
            }

            // Look up confidence
            auto cit = confidence_map.find(frame);
            bool has_conf = (cit != confidence_map.end());

            fout << frame;
            for (int k = 0; k < num_nodes; ++k) {
                fout << ",";
                if (labeled[k]) {
                    fout << px[k] << "," << py[k] << ",";
                    // confidence
                    if (has_conf && k < (int)cit->second.size() &&
                        cit->second[k] > 0.0f) {
                        fout << cit->second[k];
                    }
                    fout << ",";
                    // source flag: if has confidence, mark as Predicted
                    if (has_conf && k < (int)cit->second.size() &&
                        cit->second[k] > 0.0f) {
                        fout << "P";
                    }
                } else {
                    fout << ",,,";
                }
            }
            fout << "\n";
        }
    }

    // ---- Copy annotations.json if present ----
    {
        std::string src_json = src_folder + "/annotations.json";
        std::string dst_json = dst_folder + "/annotations.json";
        if (fs::exists(src_json)) {
            fs::copy_file(src_json, dst_json, fs::copy_options::overwrite_existing, ec);
            // Non-fatal if copy fails
        }
    }

    return true;
}

// =========================================================================
// Find most recent labels
// =========================================================================

// Search root_dir for timestamped subfolders (YYYY_MM_DD_HH_MM_SS format),
// return the most recent one. Returns 0 on success, 1 on error.
inline int find_most_recent_labels(const std::string &root_dir,
                                    std::string &most_recent_folder,
                                    std::string &error) {
    namespace fs = std::filesystem;
    std::regex datetime_regex(R"(^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$)");

    if (!fs::exists(root_dir) || !fs::is_directory(root_dir)) {
        error = "Label root directory does not exist: " + root_dir;
        return 1;
    }

    std::vector<std::string> folders;
    for (const auto &entry : fs::directory_iterator(root_dir)) {
        if (!entry.is_directory()) continue;
        std::string folder_name = entry.path().filename().string();
        if (std::regex_match(folder_name, datetime_regex)) {
            folders.push_back(entry.path().generic_string());
        }
    }

    if (folders.empty()) {
        error = "No timestamped label folders found in: " + root_dir;
        error += "\nExpected folders matching YYYY_MM_DD_HH_MM_SS format.";
        return 1;
    }

    std::sort(folders.begin(), folders.end());
    most_recent_folder = folders.back();
    return 0;
}

} // namespace AnnotationCSV

#pragma once
#include "ffmpeg_frame_reader.h"
#include "json.hpp"
#include "opencv_yaml_io.h"
#ifdef __APPLE__
#include <turbojpeg.h>
#else
#include "stb_image_write.h"
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Forward-declare AnnotationMap types (defined in annotation.h).
// Full include not needed — only used by the AnnotationMap-based overload.
#include "annotation.h"

namespace JarvisExport {

// ---------------------------------------------------------------------------
// JPEG writing — turbojpeg (SIMD-accelerated) on macOS, stb fallback elsewhere
// ---------------------------------------------------------------------------
#ifdef __APPLE__
inline bool write_jpeg(const char *path, int w, int h, int channels,
                       const uint8_t *data, int quality) {
    tjhandle tj = tjInitCompress();
    if (!tj) return false;
    unsigned char *buf = nullptr;
    unsigned long buf_size = 0;
    int pf = (channels == 3) ? TJPF_RGB : TJPF_RGBA;
    int rc = tjCompress2(tj, data, w, 0, h, pf, &buf, &buf_size,
                         TJSAMP_420, quality, TJFLAG_FASTDCT);
    bool ok = (rc == 0);
    if (ok) {
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(buf, 1, buf_size, f); fclose(f); }
        else ok = false;
    }
    tjFree(buf);
    tjDestroy(tj);
    return ok;
}
#else
inline bool write_jpeg(const char *path, int w, int h, int channels,
                       const uint8_t *data, int quality) {
    return stbi_write_jpg(path, w, h, channels, data, quality) != 0;
}
#endif

struct ExportStats {
    int train_frames = 0;
    int val_frames = 0;
    int complete_annotations = 0;
    int incomplete_annotations = 0;
    int num_cameras = 0;
    std::atomic<int> total_images_saved{0};
    double elapsed_seconds = 0.0;
    std::string output_folder;
};

struct ExportConfig {
    std::string label_folder;       // path to specific labeling session folder
    std::string calibration_folder; // path to calibration YAMLs
    std::string media_folder;       // path to video mp4s
    std::string output_folder;      // where to write output
    std::vector<std::string> camera_names;
    std::string skeleton_name;
    std::vector<std::string> node_names;
    std::vector<std::pair<int, int>> edges; // edge index pairs
    int num_keypoints = 0;
    float margin_pixel = 50.0f;
    float train_ratio = 0.9f;
    int seed = 42;
    int jpeg_quality = 95;
    int frame_start = -1; // -1 = first valid frame
    int frame_stop = -1;  // -1 = last valid frame
    int frame_step = 1;   // 1 = every frame, 10 = every 10th, etc.
    int export_num_keypoints = -1; // -1 = all, or truncate to first N keypoints
    std::string export_skeleton_name; // override skeleton name in output (empty = use source)
};

// ---------------------------------------------------------------------------
// CSV readers — match data_exporter/utils.py
// ---------------------------------------------------------------------------

// 3D CSV: Reads both v1 and v2 formats.
//   v1: skeleton_name header, then frame,idx,x,y,z,idx,x,y,z,...  (groups of 4)
//   v2: #red_csv v2 + #skeleton + column header, then frame,x,y,z,c,x,y,z,c,... (groups of 4)
// Returns map of frame_id -> Nx3 vector. Frames with any 1e7/empty sentinel are excluded.
inline std::map<int, std::vector<std::vector<double>>>
read_csv_3d(const std::string &path, int max_keypoints = -1) {
    std::map<int, std::vector<std::vector<double>>> result;
    std::ifstream file(path);
    if (!file.is_open())
        return result;

    std::string line;
    bool is_v2 = false;

    // Read first line to detect format
    if (!std::getline(file, line))
        return result;
    if (line.find("#red_csv") != std::string::npos)
        is_v2 = true;

    // Skip remaining header lines
    if (is_v2) {
        // Skip #skeleton line and column header line
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.find("frame,") == 0 || line.find("frame ") == 0) continue;
            break; // first data line
        }
        // Process this first data line, then continue loop
        if (line.empty()) return result;
        goto parse_line;
    }

    while (std::getline(file, line)) {
parse_line:
        if (line.empty())
            continue;
        // Skip any comment or header lines encountered mid-file
        if (line[0] == '#') continue;

        std::stringstream ss(line);
        std::string token;

        // frame_id
        if (!std::getline(ss, token, ','))
            continue;
        int frame_id;
        try { frame_id = std::stoi(token); }
        catch (...) { continue; }

        // Read all remaining values
        std::vector<double> values;
        while (std::getline(ss, token, ',')) {
            if (token.empty()) { values.push_back(1e7); continue; } // empty = unlabeled
            try { values.push_back(std::stod(token)); }
            catch (...) { values.push_back(1e7); }
        }

        // v1: groups of 4 (idx, x, y, z) — skip idx
        // v2: groups of 4 (x, y, z, c) — skip c
        std::vector<std::vector<double>> kps;
        if (is_v2) {
            for (size_t i = 0; i + 2 < values.size(); i += 4)
                kps.push_back({values[i], values[i+1], values[i+2]});
        } else {
            for (size_t i = 0; i + 3 < values.size(); i += 4)
                kps.push_back({values[i+1], values[i+2], values[i+3]});
        }

        // Truncate to max_keypoints if specified
        if (max_keypoints > 0 && (int)kps.size() > max_keypoints)
            kps.resize(max_keypoints);

        // Only include frame if all (truncated) keypoints are valid
        bool has_invalid = false;
        for (const auto &kp : kps) {
            if (kp[0] == 1e7 || kp[1] == 1e7 || kp[2] == 1e7) {
                has_invalid = true;
                break;
            }
        }

        if (!has_invalid)
            result[frame_id] = std::move(kps);
    }
    return result;
}

// 2D CSV: Reads both v1 and v2 formats.
//   v1: skeleton_name header, then frame,idx,x,y,idx,x,y,...  (groups of 3)
//   v2: #red_csv v2 + #skeleton + column header, then frame,x,y,c,s,x,y,c,s,...  (groups of 4)
// Applies Y-flip: y = img_height - y (converts ImPlot bottom-left to image top-left)
// Returns map of frame_id -> Nx2 vector. Unlabeled sentinels → NaN.
inline std::map<int, std::vector<std::vector<double>>>
read_csv_2d(const std::string &path, int img_height, int max_keypoints = -1) {
    std::map<int, std::vector<std::vector<double>>> result;
    std::ifstream file(path);
    if (!file.is_open())
        return result;

    std::string line;
    bool is_v2 = false;

    // Read first line to detect format
    if (!std::getline(file, line))
        return result;
    if (line.find("#red_csv") != std::string::npos)
        is_v2 = true;

    // Skip remaining header lines for v2
    if (is_v2) {
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            if (line.find("frame,") == 0 || line.find("frame ") == 0) continue;
            break; // first data line
        }
        if (line.empty()) return result;
        goto parse_line;
    }

    while (std::getline(file, line)) {
parse_line:
        if (line.empty())
            continue;
        if (line[0] == '#') continue;

        std::stringstream ss(line);
        std::string token;

        if (!std::getline(ss, token, ','))
            continue;
        int frame_id;
        try { frame_id = std::stoi(token); }
        catch (...) { continue; }

        std::vector<double> values;
        std::vector<bool> cell_empty;
        while (std::getline(ss, token, ',')) {
            if (token.empty() || token == "P" || token == "I" || token == "M") {
                values.push_back(1e7);
                cell_empty.push_back(true);
            } else {
                try {
                    values.push_back(std::stod(token));
                    cell_empty.push_back(false);
                } catch (...) {
                    values.push_back(1e7);
                    cell_empty.push_back(true);
                }
            }
        }

        std::vector<std::vector<double>> kps;
        if (is_v2) {
            // v2: groups of 4 (x, y, c, s) — extract x,y, skip c,s
            for (size_t i = 0; i + 1 < values.size(); i += 4) {
                double x = values[i];
                double y = values[i + 1];
                bool is_empty = (i < cell_empty.size() && cell_empty[i]) ||
                                (i+1 < cell_empty.size() && cell_empty[i+1]);
                if (is_empty || x == 1e7)
                    x = std::nan("");
                if (is_empty || y == 1e7)
                    y = std::nan("");
                else
                    y = img_height - y; // flip Y
                kps.push_back({x, y});
            }
        } else {
            // v1: groups of 3 (idx, x, y) — skip idx
            for (size_t i = 0; i + 2 < values.size(); i += 3) {
                double x = values[i + 1];
                double y = values[i + 2];
                if (x == 1e7)
                    x = std::nan("");
                if (y == 1e7)
                    y = std::nan("");
                else
                    y = img_height - y; // flip Y
                kps.push_back({x, y});
            }
        }
        if (max_keypoints > 0 && (int)kps.size() > max_keypoints)
            kps.resize(max_keypoints);
        result[frame_id] = std::move(kps);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Train/val split
// ---------------------------------------------------------------------------
inline void split_frames(const std::vector<int> &valid_frames, float train_ratio,
                         int seed, std::vector<int> &train_frames,
                         std::vector<int> &val_frames) {
    std::vector<int> indices(valid_frames.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    int num_train = static_cast<int>(std::floor(valid_frames.size() * train_ratio));

    std::vector<int> train_idx(indices.begin(), indices.begin() + num_train);
    std::vector<int> val_idx(indices.begin() + num_train, indices.end());

    std::sort(train_idx.begin(), train_idx.end());
    std::sort(val_idx.begin(), val_idx.end());

    train_frames.clear();
    val_frames.clear();
    for (int i : train_idx)
        train_frames.push_back(valid_frames[i]);
    for (int i : val_idx)
        val_frames.push_back(valid_frames[i]);
}

// ---------------------------------------------------------------------------
// Annotation JSON generation — matches utils.py process_one_session + generate_annotation_file
// ---------------------------------------------------------------------------
inline nlohmann::json generate_annotation_json(
    const std::string &trial_name,
    const std::vector<int> &frame_list,
    const std::string &label_folder,
    const ExportConfig &config,
    const std::map<std::string, int> &image_width,
    const std::map<std::string, int> &image_height,
    int *out_complete = nullptr,
    int *out_incomplete = nullptr) {

    nlohmann::json annotations_arr = nlohmann::json::array();
    nlohmann::json images_arr = nlohmann::json::array();
    std::map<int, std::vector<int>> set_of_frames; // frame_num -> [image_ids]

    // Load mask polygons from annotations.json (if present)
    // mask_lookup[frame_num][cam_idx] = [[x,y],[x,y],...] polygon array
    std::map<int, std::map<int, nlohmann::json>> mask_lookup;
    {
        std::string mask_path = label_folder + "/annotations.json";
        if (std::filesystem::exists(mask_path)) {
            try {
                std::ifstream mf(mask_path);
                nlohmann::json mj;
                mf >> mj;
                if (mj.contains("frames")) {
                    for (const auto &jf : mj["frames"]) {
                        int fnum = jf["frame"].get<int>();
                        if (jf.contains("cameras")) {
                            for (const auto &jc : jf["cameras"]) {
                                int c = jc["cam"].get<int>();
                                if (jc.contains("mask"))
                                    mask_lookup[fnum][c] = jc["mask"];
                            }
                        }
                    }
                }
            } catch (...) {} // non-fatal
        }
    }

    int annotation_id = 0;
    int image_id = 0;
    int n_complete = 0, n_incomplete = 0;

    // Outer loop = cameras, inner loop = frames (matches Python iteration order)
    for (int cam_idx = 0; cam_idx < (int)config.camera_names.size(); ++cam_idx) {
        const auto &cam = config.camera_names[cam_idx];
        int img_h = image_height.at(cam);
        int img_w = image_width.at(cam);

        std::string csv_path =
            label_folder + "/" + cam + ".csv";
        auto labels_2d = read_csv_2d(csv_path, img_h, config.num_keypoints);

        for (int frame_num : frame_list) {
            std::string file_name = trial_name + "/" + cam + "/Frame_" +
                                    std::to_string(frame_num) + ".jpg";

            nlohmann::json image_entry;
            image_entry["coco_url"] = "";
            image_entry["date_captured"] = "";
            image_entry["file_name"] = file_name;
            image_entry["flickr_url"] = "";
            image_entry["height"] = img_h;
            image_entry["id"] = image_id;
            image_entry["width"] = img_w;

            // Check if 2D keypoints exist and are all valid for this frame
            bool has_valid_2d = false;
            if (labels_2d.count(frame_num)) {
                const auto &kps = labels_2d[frame_num];
                bool any_nan = false;
                for (const auto &kp : kps) {
                    if (std::isnan(kp[0]) || std::isnan(kp[1])) {
                        any_nan = true;
                        break;
                    }
                }
                if (!any_nan)
                    has_valid_2d = true;
            }

            nlohmann::json annotation_entry;
            if (has_valid_2d) {
                const auto &kps = labels_2d[frame_num];

                // Compute bbox from 2D keypoints + margin
                double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
                for (const auto &kp : kps) {
                    x_min = std::min(x_min, kp[0]);
                    x_max = std::max(x_max, kp[0]);
                    y_min = std::min(y_min, kp[1]);
                    y_max = std::max(y_max, kp[1]);
                }

                x_min = std::max(x_min - config.margin_pixel, 0.0);
                x_max = std::min(x_max + config.margin_pixel, (double)img_w);
                y_min = std::max(y_min - config.margin_pixel, 0.0);
                y_max = std::min(y_max + config.margin_pixel, (double)img_h);

                double x_size = x_max - x_min;
                double y_size = y_max - y_min;

                // Flat keypoints: [x0,y0,1, x1,y1,1, ...]
                nlohmann::json keypoints_flat = nlohmann::json::array();
                for (const auto &kp : kps) {
                    keypoints_flat.push_back(static_cast<int>(kp[0]));
                    keypoints_flat.push_back(static_cast<int>(kp[1]));
                    keypoints_flat.push_back(1);
                }

                annotation_entry["bbox"] = {x_min, y_min, x_size, y_size};
                annotation_entry["category_id"] = 1;
                annotation_entry["id"] = annotation_id;
                annotation_entry["image_id"] = image_id;
                annotation_entry["iscrowd"] = 0;
                annotation_entry["keypoints"] = keypoints_flat;
                annotation_entry["num_keypoints"] = config.num_keypoints;
                // Segmentation: populate from mask_lookup if available
                {
                    nlohmann::json seg = nlohmann::json::array();
                    auto fit = mask_lookup.find(frame_num);
                    if (fit != mask_lookup.end()) {
                        auto cit = fit->second.find(cam_idx);
                        if (cit != fit->second.end()) {
                            // Flatten [[x,y],[x,y],...] → [x1,y1,x2,y2,...] (COCO format)
                            // annotations.json stores ImPlot coords (Y=0 at bottom)
                            // COCO/JARVIS needs image coords (Y=0 at top)
                            for (const auto &jpoly : cit->second) {
                                nlohmann::json flat = nlohmann::json::array();
                                for (const auto &jpt : jpoly) {
                                    flat.push_back(jpt[0].get<double>());
                                    double y = jpt[1].get<double>();
                                    flat.push_back(img_h - y); // ImPlot → image coords
                                }
                                seg.push_back(flat);
                            }
                        }
                    }
                    annotation_entry["segmentation"] = seg;
                }
            }

            // Track framesets
            set_of_frames[frame_num].push_back(image_id);

            // Append annotation only if valid
            if (has_valid_2d) {
                annotations_arr.push_back(annotation_entry);
                annotation_id++;
                n_complete++;
            } else {
                n_incomplete++;
            }

            images_arr.push_back(image_entry);
            image_id++;
        }
    }

    if (out_complete) *out_complete = n_complete;
    if (out_incomplete) *out_incomplete = n_incomplete;

    // Build skeleton array for JSON
    nlohmann::json skeleton_arr = nlohmann::json::array();
    for (size_t i = 0; i < config.edges.size(); i++) {
        nlohmann::json joint;
        joint["keypointA"] = config.node_names[config.edges[i].first];
        joint["keypointB"] = config.node_names[config.edges[i].second];
        joint["length"] = 0.0;
        joint["name"] = "Joint " + std::to_string(i + 1);
        skeleton_arr.push_back(joint);
    }

    // Categories
    nlohmann::json categories = nlohmann::json::array();
    {
        nlohmann::json cat;
        cat["id"] = 0;
        cat["name"] = config.skeleton_name;
        cat["num_keypoints"] = config.num_keypoints;
        cat["supercategory"] = "None";
        categories.push_back(cat);
    }

    // Calibrations
    nlohmann::json calib_dict;
    for (const auto &cam : config.camera_names) {
        calib_dict[cam] = "calib_params/" + trial_name + "/" + cam + ".yaml";
    }

    // Framesets
    nlohmann::json framesets;
    for (const auto &[frame_num, img_ids] : set_of_frames) {
        std::string key = trial_name + "/Frame_" + std::to_string(frame_num);
        nlohmann::json entry;
        entry["datasetName"] = trial_name;
        entry["frames"] = img_ids;
        framesets[key] = entry;
    }

    // Root JSON
    nlohmann::json root;
    root["keypoint_names"] = config.node_names;
    root["skeleton"] = skeleton_arr;
    root["categories"] = categories;
    root["annotations"] = annotations_arr;
    root["images"] = images_arr;
    root["calibrations"] = {{trial_name, calib_dict}};
    root["framesets"] = framesets;

    return root;
}

// ---------------------------------------------------------------------------
// Calibration YAML writer — transpose K, dist, R (not T)
// ---------------------------------------------------------------------------
inline bool write_calibration_yamls(const ExportConfig &config,
                                    const std::string &trial_name,
                                    const std::map<std::string, int> &image_width,
                                    const std::map<std::string, int> &image_height,
                                    std::string *status) {
    namespace fs = std::filesystem;
    std::string save_dir =
        config.output_folder + "/calib_params/" + trial_name;
    fs::create_directories(save_dir);

    for (const auto &cam : config.camera_names) {
        std::string input_path = config.calibration_folder + "/" + cam + ".yaml";

        try {
            opencv_yaml::YamlFile yaml_in = opencv_yaml::read(input_path);

            Eigen::MatrixXd K = yaml_in.getMatrix("camera_matrix");
            Eigen::MatrixXd dist = yaml_in.getMatrix("distortion_coefficients");
            Eigen::MatrixXd R = yaml_in.getMatrix("rc_ext");
            Eigen::MatrixXd T = yaml_in.getMatrix("tc_ext");

            // Transpose K, dist, R (not T) — matches Python script
            Eigen::MatrixXd Kt = K.transpose();
            Eigen::MatrixXd dist_t = dist.transpose();
            Eigen::MatrixXd Rt = R.transpose();

            std::string output_path = save_dir + "/" + cam + ".yaml";
            opencv_yaml::YamlWriter writer(output_path);
            if (!writer.isOpen()) {
                if (status)
                    *status = "Error: Cannot write calibration file: " + output_path;
                return false;
            }

            writer.writeScalar("image_width", image_width.at(cam));
            writer.writeScalar("image_height", image_height.at(cam));
            writer.writeMatrix("intrinsicMatrix", Kt);
            writer.writeMatrix("distortionCoefficients", dist_t);
            writer.writeMatrix("R", Rt);
            writer.writeMatrix("T", T);
            writer.close();
        } catch (const std::exception &e) {
            if (status)
                *status = "Error: Cannot open calibration file: " + input_path +
                          " (" + e.what() + ")";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Negative PTS offset detection
// ---------------------------------------------------------------------------
// Some MP4 videos have pre-roll frames with negative PTS before PTS=0.
// RED counts these frames during sequential playback (frame 0 = first packet),
// but frame-based seeking respects the MP4 edit list (frame 0 = PTS 0).
// This function detects the offset so we can correctly map RED frame numbers.
inline int detect_negative_pts_offset(const std::string &video_path, double fps) {
    std::string cmd = "ffprobe -v quiet -select_streams v:0 -show_packets "
                      "-show_entries packet=pts_time -of csv=p=0 \"" +
                      video_path + "\" 2>/dev/null | head -1";
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) return 0;
    char buf[256];
    int offset = 0;
    if (fgets(buf, sizeof(buf), fp)) {
        try {
            double first_pts = std::stod(buf);
            if (first_pts < -0.001) {
                offset = static_cast<int>(std::round(std::abs(first_pts) * fps));
            }
        } catch (...) {}
    }
    pclose(fp);
    return offset;
}

// ---------------------------------------------------------------------------
// JPEG frame extraction — one thread per camera
// Uses FFmpeg C API + stb_image_write instead of cv::VideoCapture + cv::imwrite
// ---------------------------------------------------------------------------
inline void extract_jpegs_for_camera(
    const std::string &cam,
    const std::string &trial_name,
    const std::string &video_path,
    const std::string &output_folder,
    const std::vector<int> &train_frames,
    const std::vector<int> &val_frames,
    const std::map<int, std::string> &frame_to_mode,
    std::string *status, std::mutex *status_mutex,
    std::atomic<int> *images_saved_counter = nullptr,
    int jpeg_quality = 95) {

    namespace fs = std::filesystem;

    // Create directories
    fs::create_directories(output_folder + "/train/" + trial_name + "/" + cam);
    fs::create_directories(output_folder + "/val/" + trial_name + "/" + cam);

    // Combine and sort all frames
    std::vector<int> all_frames;
    all_frames.insert(all_frames.end(), train_frames.begin(), train_frames.end());
    all_frames.insert(all_frames.end(), val_frames.begin(), val_frames.end());
    std::sort(all_frames.begin(), all_frames.end());

    if (all_frames.empty())
        return;

    ffmpeg_reader::FrameReader reader;
    if (!reader.open(video_path)) {
        if (status && status_mutex) {
            std::lock_guard<std::mutex> lock(*status_mutex);
            *status = "Error: Cannot open video: " + video_path;
        }
        return;
    }

    double fps = reader.fps();
    int w = reader.width();
    int h = reader.height();

    // Detect negative PTS offset: RED counts pre-roll frames with negative PTS
    // during sequential playback, but seeking respects the edit list.
    int pts_offset = detect_negative_pts_offset(video_path, fps);

    for (int frame_num : all_frames) {
        // Map RED frame number to seek frame number
        int seek_frame = frame_num - pts_offset;
        if (seek_frame < 0) continue;

        const uint8_t *rgb = reader.readFrame(seek_frame);
        if (!rgb) continue;

        auto it = frame_to_mode.find(frame_num);
        if (it != frame_to_mode.end()) {
            std::string dir = output_folder + "/" + it->second + "/" +
                              trial_name + "/" + cam + "/";
            std::string filename =
                dir + "Frame_" + std::to_string(frame_num) + ".jpg";
            write_jpeg(filename.c_str(), w, h, 3, rgb, jpeg_quality);
            if (images_saved_counter)
                images_saved_counter->fetch_add(1, std::memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// Annotation JSON generation from AnnotationMap (replaces CSV-based version)
// ---------------------------------------------------------------------------
inline nlohmann::json generate_annotation_json_from_amap(
    const std::string &trial_name,
    const std::vector<int> &frame_list,
    const AnnotationMap &amap,
    const ExportConfig &config,
    const std::map<std::string, int> &image_width,
    const std::map<std::string, int> &image_height,
    int *out_complete = nullptr,
    int *out_incomplete = nullptr) {

    nlohmann::json annotations_arr = nlohmann::json::array();
    nlohmann::json images_arr = nlohmann::json::array();
    std::map<int, std::vector<int>> set_of_frames; // frame_num -> [image_ids]

    int annotation_id = 0;
    int image_id = 0;
    int n_complete = 0, n_incomplete = 0;

    // Outer loop = cameras, inner loop = frames (matches JARVIS Python iteration order)
    for (int cam_idx = 0; cam_idx < (int)config.camera_names.size(); ++cam_idx) {
        const auto &cam_name = config.camera_names[cam_idx];
        int img_h = image_height.at(cam_name);
        int img_w = image_width.at(cam_name);

        for (int frame_num : frame_list) {
            std::string file_name = trial_name + "/" + cam_name + "/Frame_" +
                                    std::to_string(frame_num) + ".jpg";

            nlohmann::json image_entry;
            image_entry["coco_url"] = "";
            image_entry["date_captured"] = "";
            image_entry["file_name"] = file_name;
            image_entry["flickr_url"] = "";
            image_entry["height"] = img_h;
            image_entry["id"] = image_id;
            image_entry["width"] = img_w;

            // Check if all 2D keypoints are valid for this camera on this frame
            bool has_valid_2d = false;
            auto it = amap.find((u32)frame_num);
            if (it != amap.end() && cam_idx < (int)it->second.cameras.size()) {
                const auto &cam = it->second.cameras[cam_idx];
                bool any_unlabeled = false;
                for (int k = 0; k < config.num_keypoints && k < (int)cam.keypoints.size(); ++k) {
                    if (!cam.keypoints[k].labeled) { any_unlabeled = true; break; }
                }
                if (!any_unlabeled && !cam.keypoints.empty())
                    has_valid_2d = true;
            }

            nlohmann::json annotation_entry;
            if (has_valid_2d) {
                const auto &cam = it->second.cameras[cam_idx];

                // Compute bbox from 2D keypoints + margin (Y-flipped to image coords)
                double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
                for (int k = 0; k < config.num_keypoints && k < (int)cam.keypoints.size(); ++k) {
                    double x = cam.keypoints[k].x;
                    double y = img_h - cam.keypoints[k].y; // ImPlot → image coords
                    x_min = std::min(x_min, x); x_max = std::max(x_max, x);
                    y_min = std::min(y_min, y); y_max = std::max(y_max, y);
                }

                x_min = std::max(x_min - config.margin_pixel, 0.0);
                x_max = std::min(x_max + config.margin_pixel, (double)img_w);
                y_min = std::max(y_min - config.margin_pixel, 0.0);
                y_max = std::min(y_max + config.margin_pixel, (double)img_h);

                double x_size = x_max - x_min;
                double y_size = y_max - y_min;

                // Flat keypoints: [x0,y0,1, x1,y1,1, ...] — int-cast, visibility=1
                nlohmann::json keypoints_flat = nlohmann::json::array();
                for (int k = 0; k < config.num_keypoints && k < (int)cam.keypoints.size(); ++k) {
                    keypoints_flat.push_back(static_cast<int>(cam.keypoints[k].x));
                    keypoints_flat.push_back(static_cast<int>(img_h - cam.keypoints[k].y));
                    keypoints_flat.push_back(1);
                }

                // Segmentation from mask polygons (if available)
                nlohmann::json seg = nlohmann::json::array();
                if (cam.has_mask()) {
                    for (const auto &poly : cam.extras->mask_polygons) {
                        nlohmann::json flat = nlohmann::json::array();
                        for (const auto &pt : poly) {
                            flat.push_back(pt.x);
                            flat.push_back(img_h - pt.y); // ImPlot → image coords
                        }
                        seg.push_back(flat);
                    }
                }

                annotation_entry["bbox"] = {x_min, y_min, x_size, y_size};
                annotation_entry["category_id"] = 1; // JARVIS convention
                annotation_entry["id"] = annotation_id;
                annotation_entry["image_id"] = image_id;
                annotation_entry["iscrowd"] = 0;
                annotation_entry["keypoints"] = keypoints_flat;
                annotation_entry["num_keypoints"] = config.num_keypoints;
                annotation_entry["segmentation"] = seg;
            }

            // Track framesets
            set_of_frames[frame_num].push_back(image_id);

            // Append annotation only if valid
            if (has_valid_2d) {
                annotations_arr.push_back(annotation_entry);
                annotation_id++;
                n_complete++;
            } else {
                n_incomplete++;
            }

            images_arr.push_back(image_entry);
            image_id++;
        }
    }

    if (out_complete) *out_complete = n_complete;
    if (out_incomplete) *out_incomplete = n_incomplete;

    // Build skeleton array for JSON (JARVIS format: keypointA/keypointB strings)
    nlohmann::json skeleton_arr = nlohmann::json::array();
    for (size_t i = 0; i < config.edges.size(); i++) {
        nlohmann::json joint;
        joint["keypointA"] = config.node_names[config.edges[i].first];
        joint["keypointB"] = config.node_names[config.edges[i].second];
        joint["length"] = 0.0;
        joint["name"] = "Joint " + std::to_string(i + 1);
        skeleton_arr.push_back(joint);
    }

    // Categories
    nlohmann::json categories = nlohmann::json::array();
    {
        nlohmann::json cat;
        cat["id"] = 0;
        cat["name"] = config.skeleton_name;
        cat["num_keypoints"] = config.num_keypoints;
        cat["supercategory"] = "None";
        categories.push_back(cat);
    }

    // Calibrations
    nlohmann::json calib_dict;
    for (const auto &cam : config.camera_names) {
        calib_dict[cam] = "calib_params/" + trial_name + "/" + cam + ".yaml";
    }

    // Framesets
    nlohmann::json framesets;
    for (const auto &[frame_num, img_ids] : set_of_frames) {
        std::string key = trial_name + "/Frame_" + std::to_string(frame_num);
        nlohmann::json entry;
        entry["datasetName"] = trial_name;
        entry["frames"] = img_ids;
        framesets[key] = entry;
    }

    // Root JSON
    nlohmann::json root;
    root["keypoint_names"] = config.node_names;
    root["skeleton"] = skeleton_arr;
    root["categories"] = categories;
    root["annotations"] = annotations_arr;
    root["images"] = images_arr;
    root["calibrations"] = {{trial_name, calib_dict}};
    root["framesets"] = framesets;

    return root;
}

// ---------------------------------------------------------------------------
// Main export function (AnnotationMap-based)
// ---------------------------------------------------------------------------
inline bool export_jarvis_dataset(const ExportConfig &config_in,
                                  const AnnotationMap &amap,
                                  std::string *status,
                                  std::atomic<int> *images_saved_counter = nullptr) {
    namespace fs = std::filesystem;
    auto t_start = std::chrono::steady_clock::now();

    // Create timestamped subfolder inside output directory
    ExportConfig config = config_in;
    {
        time_t now = time(0);
        struct tm tstruct = *localtime(&now);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S", &tstruct);
        config.output_folder = config_in.output_folder + "/" + buf;
    }

    // Apply keypoint truncation (e.g. Rat24Target → Rat24)
    if (config.export_num_keypoints > 0 && config.export_num_keypoints < config.num_keypoints) {
        config.num_keypoints = config.export_num_keypoints;
        if ((int)config.node_names.size() > config.num_keypoints)
            config.node_names.resize(config.num_keypoints);
        std::vector<std::pair<int,int>> valid_edges;
        for (const auto &e : config.edges) {
            if (e.first < config.num_keypoints && e.second < config.num_keypoints)
                valid_edges.push_back(e);
        }
        config.edges = std::move(valid_edges);
        if (!config.export_skeleton_name.empty())
            config.skeleton_name = config.export_skeleton_name;
    }

    ExportStats stats;
    stats.output_folder = config.output_folder;
    stats.num_cameras = static_cast<int>(config.camera_names.size());

    // 1. Get valid frames from AnnotationMap (all 3D keypoints triangulated)
    std::vector<int> valid_frames;
    for (const auto &[fid, fa] : amap) {
        if (frame_is_fully_triangulated(fa, config.num_keypoints))
            valid_frames.push_back((int)fid);
    }
    std::sort(valid_frames.begin(), valid_frames.end());

    if (valid_frames.empty()) {
        if (status)
            *status = "Error: No fully-triangulated frames found";
        return false;
    }

    // Apply start/stop/step subsampling
    if (config.frame_start >= 0 || config.frame_stop >= 0 || config.frame_step > 1) {
        int start = (config.frame_start >= 0) ? config.frame_start : valid_frames.front();
        int stop  = (config.frame_stop >= 0)  ? config.frame_stop  : valid_frames.back();
        int step  = std::max(1, config.frame_step);
        std::vector<int> subsampled;
        for (int f : valid_frames) {
            if (f < start) continue;
            if (f > stop) break;
            subsampled.push_back(f);
        }
        if (step > 1) {
            std::vector<int> stepped;
            for (size_t i = 0; i < subsampled.size(); i += step)
                stepped.push_back(subsampled[i]);
            subsampled = std::move(stepped);
        }
        valid_frames = std::move(subsampled);
        if (valid_frames.empty()) {
            if (status)
                *status = "Error: No frames in specified range";
            return false;
        }
    }

    // 2. Train/val split
    std::vector<int> train_frames, val_frames;
    split_frames(valid_frames, config.train_ratio, config.seed, train_frames,
                 val_frames);

    stats.train_frames = static_cast<int>(train_frames.size());
    stats.val_frames = static_cast<int>(val_frames.size());

    // 3. Read image dimensions from calibration files
    std::map<std::string, int> image_width, image_height;
    for (const auto &cam : config.camera_names) {
        std::string calib_path =
            config.calibration_folder + "/" + cam + ".yaml";
        try {
            opencv_yaml::YamlFile yaml = opencv_yaml::read(calib_path);
            image_width[cam] = yaml.getInt("image_width");
            image_height[cam] = yaml.getInt("image_height");
        } catch (const std::exception &e) {
            if (status)
                *status = "Error: Cannot open calibration: " + calib_path;
            return false;
        }
    }

    // Extract trial_name from label_folder (last directory component)
    std::string trial_name = fs::path(config.label_folder).filename().string();
    if (trial_name.empty()) trial_name = "export";

    // 4. Generate train annotation JSON from AnnotationMap
    int train_complete = 0, train_incomplete = 0;
    auto train_json = generate_annotation_json_from_amap(
        trial_name, train_frames, amap, config, image_width,
        image_height, &train_complete, &train_incomplete);

    // 5. Generate val annotation JSON from AnnotationMap
    int val_complete = 0, val_incomplete = 0;
    auto val_json = generate_annotation_json_from_amap(
        trial_name, val_frames, amap, config, image_width,
        image_height, &val_complete, &val_incomplete);

    stats.complete_annotations = train_complete + val_complete;
    stats.incomplete_annotations = train_incomplete + val_incomplete;

    // 6. Write annotation JSONs
    fs::create_directories(config.output_folder + "/annotations");

    {
        std::ofstream f(config.output_folder + "/annotations/instances_train.json");
        if (!f.is_open()) {
            if (status)
                *status = "Error: Cannot write instances_train.json";
            return false;
        }
        f << train_json.dump(4);
    }
    {
        std::ofstream f(config.output_folder + "/annotations/instances_val.json");
        if (!f.is_open()) {
            if (status)
                *status = "Error: Cannot write instances_val.json";
            return false;
        }
        f << val_json.dump(4);
    }

    // 7. Write calibration YAMLs
    if (!write_calibration_yamls(config, trial_name, image_width, image_height,
                                 status))
        return false;

    // 8. Extract JPEG frames — one thread per camera
    std::map<int, std::string> frame_to_mode;
    for (int f : train_frames)
        frame_to_mode[f] = "train";
    for (int f : val_frames)
        frame_to_mode[f] = "val";

    // Use external counter for progress bar if provided, else internal
    std::atomic<int> *counter = images_saved_counter
        ? images_saved_counter : &stats.total_images_saved;

    std::mutex status_mutex;
    std::vector<std::thread> threads;
    for (const auto &cam : config.camera_names) {
        std::string video_path =
            config.media_folder + "/" + cam + ".mp4";
        threads.emplace_back(extract_jpegs_for_camera, cam, trial_name,
                             video_path, config.output_folder, train_frames,
                             val_frames, frame_to_mode, status, &status_mutex,
                             counter, config.jpeg_quality);
    }
    for (auto &t : threads)
        t.join();
    if (images_saved_counter)
        stats.total_images_saved.store(images_saved_counter->load());

    if (status && status->find("Error") != std::string::npos)
        return false;

    auto t_end = std::chrono::steady_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    int total_frames = stats.train_frames + stats.val_frames;
    std::cout << "\n=== JARVIS Export Complete ===" << std::endl;
    std::cout << "Output:      " << stats.output_folder << std::endl;
    std::cout << "Frames:      " << stats.train_frames << " train, "
              << stats.val_frames << " val (" << total_frames << " total)" << std::endl;
    std::cout << "Images:      " << stats.total_images_saved.load() << " saved ("
              << stats.num_cameras << " cameras x " << total_frames << " frames)" << std::endl;
    std::cout << "Annotations: " << stats.complete_annotations << " complete, "
              << stats.incomplete_annotations << " incomplete" << std::endl;
    std::cout << "Duration:    " << std::fixed << std::setprecision(1)
              << stats.elapsed_seconds << "s" << std::endl;
    std::cout << std::endl;

    if (status)
        *status = "Export completed successfully! Train: " +
                  std::to_string(train_frames.size()) +
                  ", Val: " + std::to_string(val_frames.size()) +
                  " frames x " + std::to_string(config.camera_names.size()) +
                  " cameras";
    return true;
}

// ---------------------------------------------------------------------------
// Main export function (legacy CSV-based — kept for backward compatibility)
// ---------------------------------------------------------------------------
inline bool export_jarvis_dataset(const ExportConfig &config_in,
                                  std::string *status) {
    namespace fs = std::filesystem;
    auto t_start = std::chrono::steady_clock::now();

    // Create timestamped subfolder inside output directory
    ExportConfig config = config_in;
    {
        time_t now = time(0);
        struct tm tstruct = *localtime(&now);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S", &tstruct);
        config.output_folder = config_in.output_folder + "/" + buf;
    }

    // Apply keypoint truncation (e.g. Rat24Target → Rat24: drop last keypoint)
    if (config.export_num_keypoints > 0 && config.export_num_keypoints < config.num_keypoints) {
        config.num_keypoints = config.export_num_keypoints;
        if ((int)config.node_names.size() > config.num_keypoints)
            config.node_names.resize(config.num_keypoints);
        // Keep only edges that reference valid keypoint indices
        std::vector<std::pair<int,int>> valid_edges;
        for (const auto &e : config.edges) {
            if (e.first < config.num_keypoints && e.second < config.num_keypoints)
                valid_edges.push_back(e);
        }
        config.edges = std::move(valid_edges);
        if (!config.export_skeleton_name.empty())
            config.skeleton_name = config.export_skeleton_name;
    }

    ExportStats stats;
    stats.output_folder = config.output_folder;
    stats.num_cameras = static_cast<int>(config.camera_names.size());

    if (status)
        *status = "Reading 3D keypoints...";

    // 1. Read 3D CSV to get valid frames (truncated to export_num_keypoints)
    std::string kp3d_path = config.label_folder + "/keypoints3d.csv";
    auto labels_3d = read_csv_3d(kp3d_path, config.num_keypoints);
    if (labels_3d.empty()) {
        if (status)
            *status = "Error: No valid 3D keypoints found in " + kp3d_path;
        return false;
    }

    // Collect valid frame indices (sorted)
    std::vector<int> valid_frames;
    for (const auto &[fid, _] : labels_3d)
        valid_frames.push_back(fid);
    std::sort(valid_frames.begin(), valid_frames.end());

    // Apply start/stop/step subsampling
    if (config.frame_start >= 0 || config.frame_stop >= 0 || config.frame_step > 1) {
        int start = (config.frame_start >= 0) ? config.frame_start : valid_frames.front();
        int stop  = (config.frame_stop >= 0)  ? config.frame_stop  : valid_frames.back();
        int step  = std::max(1, config.frame_step);
        std::vector<int> subsampled;
        for (int f : valid_frames) {
            if (f < start) continue;
            if (f > stop) break;
            subsampled.push_back(f);
        }
        if (step > 1) {
            std::vector<int> stepped;
            for (size_t i = 0; i < subsampled.size(); i += step)
                stepped.push_back(subsampled[i]);
            subsampled = std::move(stepped);
        }
        valid_frames = std::move(subsampled);
    }

    if (valid_frames.empty()) {
        if (status)
            *status = "Error: No frames in specified range";
        return false;
    }

    if (status)
        *status = "Splitting " + std::to_string(valid_frames.size()) +
                  " frames into train/val...";

    // 2. Train/val split
    std::vector<int> train_frames, val_frames;
    split_frames(valid_frames, config.train_ratio, config.seed, train_frames,
                 val_frames);

    stats.train_frames = static_cast<int>(train_frames.size());
    stats.val_frames = static_cast<int>(val_frames.size());

    // 3. Read image dimensions from calibration files
    std::map<std::string, int> image_width, image_height;
    for (const auto &cam : config.camera_names) {
        std::string calib_path =
            config.calibration_folder + "/" + cam + ".yaml";
        try {
            opencv_yaml::YamlFile yaml = opencv_yaml::read(calib_path);
            image_width[cam] = yaml.getInt("image_width");
            image_height[cam] = yaml.getInt("image_height");
        } catch (const std::exception &e) {
            if (status)
                *status = "Error: Cannot open calibration: " + calib_path;
            return false;
        }
    }

    // Extract trial_name from label_folder (last directory component)
    std::string trial_name = fs::path(config.label_folder).filename().string();

    // 4. Generate train annotation JSON
    if (status)
        *status = "Generating train annotations...";
    int train_complete = 0, train_incomplete = 0;
    auto train_json = generate_annotation_json(
        trial_name, train_frames, config.label_folder, config, image_width,
        image_height, &train_complete, &train_incomplete);

    // 5. Generate val annotation JSON
    if (status)
        *status = "Generating val annotations...";
    int val_complete = 0, val_incomplete = 0;
    auto val_json = generate_annotation_json(
        trial_name, val_frames, config.label_folder, config, image_width,
        image_height, &val_complete, &val_incomplete);

    stats.complete_annotations = train_complete + val_complete;
    stats.incomplete_annotations = train_incomplete + val_incomplete;

    // 6. Write annotation JSONs
    if (status)
        *status = "Writing annotation files...";
    fs::create_directories(config.output_folder + "/annotations");

    {
        std::ofstream f(config.output_folder + "/annotations/instances_train.json");
        if (!f.is_open()) {
            if (status)
                *status = "Error: Cannot write instances_train.json";
            return false;
        }
        f << train_json.dump(4);
    }
    {
        std::ofstream f(config.output_folder + "/annotations/instances_val.json");
        if (!f.is_open()) {
            if (status)
                *status = "Error: Cannot write instances_val.json";
            return false;
        }
        f << val_json.dump(4);
    }

    // 7. Write calibration YAMLs
    if (status)
        *status = "Writing calibration files...";
    if (!write_calibration_yamls(config, trial_name, image_width, image_height,
                                 status))
        return false;

    // 8. Extract JPEG frames — one thread per camera
    if (status)
        *status = "Extracting JPEG frames (" +
                  std::to_string(config.camera_names.size()) + " cameras)...";

    // Build frame→mode map
    std::map<int, std::string> frame_to_mode;
    for (int f : train_frames)
        frame_to_mode[f] = "train";
    for (int f : val_frames)
        frame_to_mode[f] = "val";

    std::mutex status_mutex;
    std::vector<std::thread> threads;
    for (const auto &cam : config.camera_names) {
        std::string video_path =
            config.media_folder + "/" + cam + ".mp4";
        threads.emplace_back(extract_jpegs_for_camera, cam, trial_name,
                             video_path, config.output_folder, train_frames,
                             val_frames, frame_to_mode, status, &status_mutex,
                             &stats.total_images_saved, config.jpeg_quality);
    }
    for (auto &t : threads)
        t.join();

    // Check if status was set to an error during extraction
    if (status && status->find("Error") != std::string::npos)
        return false;

    auto t_end = std::chrono::steady_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();

    // Print export summary
    int total_frames = stats.train_frames + stats.val_frames;
    std::cout << "\n=== JARVIS Export Complete ===" << std::endl;
    std::cout << "Output:      " << stats.output_folder << std::endl;
    std::cout << "Frames:      " << stats.train_frames << " train, "
              << stats.val_frames << " val (" << total_frames << " total)" << std::endl;
    std::cout << "Images:      " << stats.total_images_saved.load() << " saved ("
              << stats.num_cameras << " cameras x " << total_frames << " frames)" << std::endl;
    std::cout << "Annotations: " << stats.complete_annotations << " complete, "
              << stats.incomplete_annotations << " incomplete" << std::endl;
    std::cout << "Duration:    " << std::fixed << std::setprecision(1)
              << stats.elapsed_seconds << "s" << std::endl;
    std::cout << std::endl;

    if (status)
        *status = "Export completed successfully! Train: " +
                  std::to_string(train_frames.size()) +
                  ", Val: " + std::to_string(val_frames.size()) +
                  " frames x " + std::to_string(config.camera_names.size()) +
                  " cameras";
    return true;
}

} // namespace JarvisExport

#pragma once
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace JarvisExport {

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
};

// ---------------------------------------------------------------------------
// CSV readers — match data_exporter/utils.py
// ---------------------------------------------------------------------------

// 3D CSV: header line (skeleton name), then frame_id, node_idx, x, y, z, ...
// Returns map of frame_id -> Nx3 vector. Frames with any 1e7 sentinel are excluded.
inline std::map<int, std::vector<std::vector<double>>>
read_csv_3d(const std::string &path) {
    std::map<int, std::vector<std::vector<double>>> result;
    std::ifstream file(path);
    if (!file.is_open())
        return result;

    std::string line;
    // skip header
    if (!std::getline(file, line))
        return result;

    while (std::getline(file, line)) {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string token;

        // frame_id
        if (!std::getline(ss, token, ','))
            continue;
        int frame_id = std::stoi(token);

        // remaining: node_idx, x, y, z, node_idx, x, y, z, ...
        std::vector<double> values;
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stod(token));
        }

        // group into [node_idx, x, y, z] quads, keep only x,y,z
        std::vector<std::vector<double>> kps;
        bool has_invalid = false;
        for (size_t i = 0; i + 3 < values.size(); i += 4) {
            double x = values[i + 1];
            double y = values[i + 2];
            double z = values[i + 3];
            if (x == 1e7 || y == 1e7 || z == 1e7)
                has_invalid = true;
            kps.push_back({x, y, z});
        }

        if (!has_invalid)
            result[frame_id] = std::move(kps);
    }
    return result;
}

// 2D CSV: header line, then frame_id, node_idx, x, y, node_idx, x, y, ...
// Applies Y-flip: y = img_height - y (converts ImPlot bottom-left to image top-left)
// Returns map of frame_id -> Nx2 vector. 1e7 sentinels → NaN.
inline std::map<int, std::vector<std::vector<double>>>
read_csv_2d(const std::string &path, int img_height) {
    std::map<int, std::vector<std::vector<double>>> result;
    std::ifstream file(path);
    if (!file.is_open())
        return result;

    std::string line;
    // skip header
    if (!std::getline(file, line))
        return result;

    while (std::getline(file, line)) {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string token;

        if (!std::getline(ss, token, ','))
            continue;
        int frame_id = std::stoi(token);

        std::vector<double> values;
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stod(token));
        }

        // group into [node_idx, x, y] triples, keep only x,y
        std::vector<std::vector<double>> kps;
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
    const std::map<std::string, int> &image_height) {

    nlohmann::json annotations_arr = nlohmann::json::array();
    nlohmann::json images_arr = nlohmann::json::array();
    std::map<int, std::vector<int>> set_of_frames; // frame_num -> [image_ids]

    int annotation_id = 0;
    int image_id = 0;

    // Outer loop = cameras, inner loop = frames (matches Python iteration order)
    for (const auto &cam : config.camera_names) {
        int img_h = image_height.at(cam);
        int img_w = image_width.at(cam);

        std::string csv_path =
            label_folder + "/" + cam + ".csv";
        auto labels_2d = read_csv_2d(csv_path, img_h);

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
                annotation_entry["segmentation"] = nlohmann::json::array();
            }

            // Track framesets
            set_of_frames[frame_num].push_back(image_id);

            // Append annotation only if valid
            if (has_valid_2d) {
                annotations_arr.push_back(annotation_entry);
                annotation_id++;
            }

            images_arr.push_back(image_entry);
            image_id++;
        }
    }

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
        cv::FileStorage fs_in(input_path, cv::FileStorage::READ);
        if (!fs_in.isOpened()) {
            if (status)
                *status = "Error: Cannot open calibration file: " + input_path;
            return false;
        }

        cv::Mat K, dist, R, T;
        fs_in["camera_matrix"] >> K;
        fs_in["distortion_coefficients"] >> dist;
        fs_in["rc_ext"] >> R;
        fs_in["tc_ext"] >> T;

        // Transpose K, dist, R (not T) — matches Python script
        K = K.t();
        dist = dist.t();
        R = R.t();

        std::string output_path = save_dir + "/" + cam + ".yaml";
        cv::FileStorage fs_out(output_path, cv::FileStorage::WRITE);
        if (!fs_out.isOpened()) {
            if (status)
                *status = "Error: Cannot write calibration file: " + output_path;
            return false;
        }

        fs_out.write("image_width", image_width.at(cam));
        fs_out.write("image_height", image_height.at(cam));
        fs_out.write("intrinsicMatrix", K);
        fs_out.write("distortionCoefficients", dist);
        fs_out.write("R", R);
        fs_out.write("T", T);
        fs_out.release();
    }
    return true;
}

// ---------------------------------------------------------------------------
// JPEG frame extraction — one thread per camera
// ---------------------------------------------------------------------------
inline void extract_jpegs_for_camera(
    const std::string &cam,
    const std::string &trial_name,
    const std::string &video_path,
    const std::string &output_folder,
    const std::vector<int> &train_frames,
    const std::vector<int> &val_frames,
    const std::map<int, std::string> &frame_to_mode,
    std::string *status, std::mutex *status_mutex) {

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

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        if (status && status_mutex) {
            std::lock_guard<std::mutex> lock(*status_mutex);
            *status = "Error: Cannot open video: " + video_path;
        }
        return;
    }

    // Seek-and-read: for each needed frame, either read sequentially if close
    // to current position, or seek directly if there's a large gap.
    // This avoids decoding thousands of intermediate frames.
    constexpr int SEEK_THRESHOLD = 30; // roughly one keyframe interval

    int current_pos = -1;
    for (size_t i = 0; i < all_frames.size(); i++) {
        int target = all_frames[i];

        if (current_pos < 0 || target - current_pos > SEEK_THRESHOLD) {
            // Seek directly to target frame
            cap.set(cv::CAP_PROP_POS_FRAMES, target);
            current_pos = target;
        } else {
            // Read sequentially, skipping unneeded frames
            while (current_pos < target) {
                cap.grab(); // decode but don't retrieve — faster than read()
                current_pos++;
            }
        }

        cv::Mat frame;
        if (!cap.read(frame))
            break;
        current_pos++;

        auto it = frame_to_mode.find(target);
        if (it != frame_to_mode.end()) {
            std::string dir = output_folder + "/" + it->second + "/" +
                              trial_name + "/" + cam + "/";
            std::string filename =
                dir + "Frame_" + std::to_string(target) + ".jpg";
            cv::imwrite(filename, frame);
        }
    }
    cap.release();
}

// ---------------------------------------------------------------------------
// Main export function
// ---------------------------------------------------------------------------
inline bool export_jarvis_dataset(const ExportConfig &config,
                                  std::string *status) {
    namespace fs = std::filesystem;

    if (status)
        *status = "Reading 3D keypoints...";

    // 1. Read 3D CSV to get valid frames
    std::string kp3d_path = config.label_folder + "/keypoints3d.csv";
    auto labels_3d = read_csv_3d(kp3d_path);
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

    if (status)
        *status = "Splitting " + std::to_string(valid_frames.size()) +
                  " frames into train/val...";

    // 2. Train/val split
    std::vector<int> train_frames, val_frames;
    split_frames(valid_frames, config.train_ratio, config.seed, train_frames,
                 val_frames);

    // 3. Read image dimensions from calibration files
    std::map<std::string, int> image_width, image_height;
    for (const auto &cam : config.camera_names) {
        std::string calib_path =
            config.calibration_folder + "/" + cam + ".yaml";
        cv::FileStorage fs(calib_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            if (status)
                *status = "Error: Cannot open calibration: " + calib_path;
            return false;
        }
        image_width[cam] = static_cast<int>(fs["image_width"].real());
        image_height[cam] = static_cast<int>(fs["image_height"].real());
    }

    // Extract trial_name from label_folder (last directory component)
    std::string trial_name = fs::path(config.label_folder).filename().string();

    // 4. Generate train annotation JSON
    if (status)
        *status = "Generating train annotations...";
    auto train_json = generate_annotation_json(
        trial_name, train_frames, config.label_folder, config, image_width,
        image_height);

    // 5. Generate val annotation JSON
    if (status)
        *status = "Generating val annotations...";
    auto val_json = generate_annotation_json(
        trial_name, val_frames, config.label_folder, config, image_width,
        image_height);

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
                             val_frames, frame_to_mode, status, &status_mutex);
    }
    for (auto &t : threads)
        t.join();

    // Check if status was set to an error during extraction
    if (status && status->find("Error") != std::string::npos)
        return false;

    if (status)
        *status = "Export completed successfully! Train: " +
                  std::to_string(train_frames.size()) +
                  ", Val: " + std::to_string(val_frames.size()) +
                  " frames x " + std::to_string(config.camera_names.size()) +
                  " cameras";
    return true;
}

} // namespace JarvisExport

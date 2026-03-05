#pragma once
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace CalibrationTool {

struct CharucoSetup {
    int w = 5, h = 5;
    float square_side_length = 60.0f;
    float marker_side_length = 37.5f;
    int dictionary = 0;
};

struct CalibConfig {
    std::string img_path;
    std::string vid_path;
    std::vector<std::string> cam_ordered;
    CharucoSetup charuco_setup;
    int first_view = 0;
    std::vector<int> second_view_order;
    std::string global_registration_video;
    std::vector<std::string> world_coordinate_imgs;
    std::map<std::string, std::vector<std::vector<float>>> gt_pts;
    nlohmann::json ba_config;
};

// Parse config.json into CalibConfig. Returns true on success.
inline bool parse_config(const std::string &config_path,
                         CalibConfig &config, std::string &error) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        error = "Cannot open config file: " + config_path;
        return false;
    }

    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception &e) {
        error = std::string("JSON parse error: ") + e.what();
        return false;
    }

    config.img_path = j.value("img_path", std::string{});
    config.vid_path = j.value("vid_path", std::string{});
    config.global_registration_video =
        j.value("global_registration_video", std::string{});

    if (j.contains("cam_ordered") && j["cam_ordered"].is_array()) {
        config.cam_ordered = j["cam_ordered"].get<std::vector<std::string>>();
    }
    if (config.cam_ordered.empty()) {
        error = "config.json has no cam_ordered array";
        return false;
    }

    if (j.contains("charuco_setup") && j["charuco_setup"].is_object()) {
        auto &cs = j["charuco_setup"];
        config.charuco_setup.w = cs.value("w", 5);
        config.charuco_setup.h = cs.value("h", 5);
        config.charuco_setup.square_side_length =
            cs.value("square_side_length", 60.0f);
        config.charuco_setup.marker_side_length =
            cs.value("marker_side_length", 37.5f);
        config.charuco_setup.dictionary = cs.value("dictionary", 0);
    }

    config.first_view = j.value("first_view", 0);

    if (j.contains("second_view_order") && j["second_view_order"].is_array()) {
        config.second_view_order =
            j["second_view_order"].get<std::vector<int>>();
    }

    if (j.contains("world_coordinate_imgs") &&
        j["world_coordinate_imgs"].is_array()) {
        config.world_coordinate_imgs =
            j["world_coordinate_imgs"].get<std::vector<std::string>>();
    }

    if (j.contains("gt_pts") && j["gt_pts"].is_object()) {
        for (auto &[key, val] : j["gt_pts"].items()) {
            std::vector<std::vector<float>> pts;
            for (auto &pt : val) {
                pts.push_back(pt.get<std::vector<float>>());
            }
            config.gt_pts[key] = std::move(pts);
        }
    }

    if (j.contains("ba_config")) {
        config.ba_config = j["ba_config"];
    }

    return true;
}

// Scan img_path for images matching cam_ordered serials.
// Returns map: {"serial_N.jpg" -> "/full/path/serial_N.jpg"}
// This is the format load_images() expects.
inline std::map<std::string, std::string>
discover_images(const CalibConfig &config) {
    namespace fs = std::filesystem;
    std::map<std::string, std::string> result;

    if (config.img_path.empty() || !fs::is_directory(config.img_path))
        return result;

    // Build set of known serials for fast lookup
    std::set<std::string> serials(config.cam_ordered.begin(),
                                  config.cam_ordered.end());

    for (const auto &entry : fs::directory_iterator(config.img_path)) {
        if (!entry.is_regular_file())
            continue;
        std::string filename = entry.path().filename().string();

        // Check extension
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
            ext != ".tiff" && ext != ".tif")
            continue;

        // Extract serial: everything before first underscore
        auto underscore = filename.find('_');
        if (underscore == std::string::npos)
            continue;
        std::string serial = filename.substr(0, underscore);

        if (serials.count(serial)) {
            result[filename] = entry.path().string();
        }
    }

    return result;
}

// Count images per camera from a discovered file map.
inline int count_images_per_camera(
    const std::map<std::string, std::string> &files,
    const std::string &serial) {
    int count = 0;
    std::string prefix = serial + "_";
    for (const auto &[filename, _] : files) {
        if (filename.substr(0, prefix.size()) == prefix)
            count++;
    }
    return count;
}

// Get the default output folder for calibration (parent of config file + /calibration).
inline std::string
default_output_folder(const std::string &config_path) {
    auto parent = std::filesystem::path(config_path).parent_path();
    return (parent / "calibration").string();
}

} // namespace CalibrationTool

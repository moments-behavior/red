#pragma once
#include "json.hpp"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

struct UserSettings {
    // Paths
    std::string default_project_root_path;
    std::string default_media_root_path;

    // Display defaults
    int default_brightness = 0;
    float default_contrast = 1.0f;
    bool default_pivot_midgray = true;

    // Playback defaults
    float default_playback_speed = 1.0f;
    bool default_realtime_playback = true;
    int default_buffer_size = 64;

    // Hardware (Linux/Windows only — Mac always uses host buffer).
    // CPU Buffer is the default because the display ring buffer scales
    // multiplicatively (cams × buffer_size × frame_bytes) and quickly
    // exceeds GPU memory on multi-cam high-res setups (16 cams × 24
    // slots × 28 MB ≈ 10 GiB just for display). Under CPU Buffer that
    // lives on host where 10 GiB of RAM is cheap.
    //
    // GPU Buffer is the architecturally faster path (no per-frame H2D
    // for render, no D2H during decode, enables HN's device-side
    // preprocessing kernels for ~120 ms savings per predict) but
    // requires the user to reduce buffer_size to fit the model + frame
    // ring within GPU memory. Changes take effect on the next launch
    // (buffers are allocated at startup).
    bool use_cpu_buffer = true;

    // HybridNet 3D prediction: number of (spread) cameras used for CenterDetect
    // to locate the crop ROI. Keypoint detection always uses all cameras; fewer
    // center cams ≈ halves the center stage at negligible accuracy cost. Clamped
    // to [2, num_cameras] at use; set == num_cameras to use all.
    int jarvis_center_cams = 8;

    // Keypoint colors. keypoint_colormap: -1 == legacy HSV rainbow (default),
    // >= 0 == an ImPlotColormap_ index (Viridis, Plasma, Jet, ...).
    // active_keypoint_color: RGB highlight for the selected keypoint (default
    // white), shown in the camera views and the Keypoints table.
    int keypoint_colormap = -1;
    std::vector<float> active_keypoint_color = {1.0f, 1.0f, 1.0f};

    // Export defaults
    float jarvis_margin = 50.0f;
    float jarvis_train_ratio = 0.9f;
    int jarvis_seed = 42;
    int jarvis_jpeg_quality = 95;

    // MuJoCo
    std::string last_mujoco_model;

    // Arena alignment corners (4 corners x 3 coords, in calibration mm)
    std::vector<double> arena_corners; // 12 values, empty if not set

    // Recent projects (most recent first, max 10)
    std::vector<std::string> recent_projects;

    void push_recent_project(const std::string &path) {
        if (path.empty()) return;
        // Remove existing entry if present (dedup)
        recent_projects.erase(
            std::remove(recent_projects.begin(), recent_projects.end(), path),
            recent_projects.end());
        // Insert at front
        recent_projects.insert(recent_projects.begin(), path);
        // Cap at 10
        if (recent_projects.size() > 10)
            recent_projects.resize(10);
    }
};

inline void to_json(nlohmann::json &j, const UserSettings &s) {
    j = nlohmann::json{
        {"default_project_root_path", s.default_project_root_path},
        {"default_media_root_path", s.default_media_root_path},
        {"default_brightness", s.default_brightness},
        {"default_contrast", s.default_contrast},
        {"default_pivot_midgray", s.default_pivot_midgray},
        {"default_playback_speed", s.default_playback_speed},
        {"default_realtime_playback", s.default_realtime_playback},
        {"default_buffer_size", s.default_buffer_size},
        {"use_cpu_buffer", s.use_cpu_buffer},
        {"keypoint_colormap", s.keypoint_colormap},
        {"active_keypoint_color", s.active_keypoint_color},
        {"jarvis_center_cams", s.jarvis_center_cams},
        {"jarvis_margin", s.jarvis_margin},
        {"jarvis_train_ratio", s.jarvis_train_ratio},
        {"jarvis_seed", s.jarvis_seed},
        {"jarvis_jpeg_quality", s.jarvis_jpeg_quality},
        {"last_mujoco_model", s.last_mujoco_model},
        {"arena_corners", s.arena_corners},
        {"recent_projects", s.recent_projects}};
}

inline void from_json(const nlohmann::json &j, UserSettings &s) {
    s.default_project_root_path =
        j.value("default_project_root_path", std::string{});
    s.default_media_root_path =
        j.value("default_media_root_path", std::string{});
    s.default_brightness = j.value("default_brightness", 0);
    s.default_contrast = j.value("default_contrast", 1.0f);
    s.default_pivot_midgray = j.value("default_pivot_midgray", true);
    s.default_playback_speed = j.value("default_playback_speed", 1.0f);
    s.default_realtime_playback = j.value("default_realtime_playback", true);
    s.default_buffer_size = j.value("default_buffer_size", 64);
    s.use_cpu_buffer = j.value("use_cpu_buffer", true);
    s.keypoint_colormap = j.value("keypoint_colormap", -1);
    s.active_keypoint_color =
        j.value("active_keypoint_color", std::vector<float>{1.0f, 1.0f, 1.0f});
    s.jarvis_center_cams = j.value("jarvis_center_cams", 8);
    s.jarvis_margin = j.value("jarvis_margin", 50.0f);
    s.jarvis_train_ratio = j.value("jarvis_train_ratio", 0.9f);
    s.jarvis_seed = j.value("jarvis_seed", 42);
    s.jarvis_jpeg_quality = j.value("jarvis_jpeg_quality", 95);
    s.last_mujoco_model = j.value("last_mujoco_model", std::string{});
    s.arena_corners = j.value("arena_corners", std::vector<double>{});
    s.recent_projects = j.value("recent_projects", std::vector<std::string>{});
}

inline std::filesystem::path user_settings_path() {
    return std::filesystem::path(get_home_directory()) /
           ".config/red/user_settings.json";
}

inline UserSettings load_user_settings() {
    UserSettings s;
    std::filesystem::path p = user_settings_path();
    if (!std::filesystem::exists(p))
        return s;
    try {
        std::ifstream f(p);
        nlohmann::json j;
        f >> j;
        s = j.get<UserSettings>();
    } catch (const std::exception &e) {
        std::cerr << "Failed to read user_settings.json: " << e.what()
                  << std::endl;
    }
    return s;
}

inline bool save_user_settings(const UserSettings &s) {
    std::filesystem::path p = user_settings_path();
    try {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            std::cerr << "Cannot create settings dir: " << ec.message()
                      << std::endl;
            return false;
        }
        std::ofstream f(p, std::ios::binary);
        if (!f)
            return false;
        nlohmann::json j = s;
        f << j.dump(2);
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Failed to write user_settings.json: " << e.what()
                  << std::endl;
        return false;
    }
}

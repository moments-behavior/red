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

    // Export defaults
    float jarvis_margin = 50.0f;
    float jarvis_train_ratio = 0.9f;
    int jarvis_seed = 42;
    int jarvis_jpeg_quality = 95;

    // MuJoCo
    std::string last_mujoco_model;

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
        {"jarvis_margin", s.jarvis_margin},
        {"jarvis_train_ratio", s.jarvis_train_ratio},
        {"jarvis_seed", s.jarvis_seed},
        {"jarvis_jpeg_quality", s.jarvis_jpeg_quality},
        {"last_mujoco_model", s.last_mujoco_model},
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
    s.jarvis_margin = j.value("jarvis_margin", 50.0f);
    s.jarvis_train_ratio = j.value("jarvis_train_ratio", 0.9f);
    s.jarvis_seed = j.value("jarvis_seed", 42);
    s.jarvis_jpeg_quality = j.value("jarvis_jpeg_quality", 95);
    s.last_mujoco_model = j.value("last_mujoco_model", std::string{});
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
